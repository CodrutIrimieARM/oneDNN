/*******************************************************************************
* Copyright 2026 Arm Ltd. and affiliates
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cassert>

#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/aarch64/prelu/jit_uni_prelu_forward.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

namespace prelu {

static bool dims_equal(
        const dims_t &lhs_dims, const dims_t &rhs_dims, dim_t ndims) {
    for (dim_t i = 0; i < ndims; ++i)
        if (lhs_dims[i] != rhs_dims[i]) return false;
    return true;
}

static bool is_full_bcast(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d) {
    const auto src_ndims = src_d.ndims();

    // Full broadcast here means "not broadcast at all": weights match src
    // element-for-element, including the physical blocking information.
    if (src_ndims != weights_d.ndims()) return false;
    if (!dims_equal(src_d.dims(), weights_d.dims(), src_ndims)) return false;
    if (src_d.format_kind() != weights_d.format_kind()) return false;

    if (!src_d.is_blocking_desc()) return true;

    const auto &src_bd = src_d.blocking_desc();
    const auto &weights_bd = weights_d.blocking_desc();

    return src_bd.inner_nblks == weights_bd.inner_nblks
            && dims_equal(src_bd.strides, weights_bd.strides, src_d.ndims())
            && dims_equal(src_bd.inner_blks, weights_bd.inner_blks,
                    src_bd.inner_nblks)
            && dims_equal(src_bd.inner_idxs, weights_bd.inner_idxs,
                    src_bd.inner_nblks);
}

static bool is_per_oc_bcast(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d) {
    const auto ndims = src_d.ndims();
    if (weights_d.ndims() != ndims || ndims < 2) return false;

    const auto &src_dims = src_d.dims();
    const auto &weights_dims = weights_d.dims();

    // PReLU's common channel-wise form: {1, C, 1, 1, ...}. The one slope value
    // for each channel is reused across minibatch and spatial dimensions.
    bool ok = weights_dims[0] == 1 && weights_dims[1] == src_dims[1];
    for (int d = 2; ok && d < ndims; ++d)
        ok = weights_dims[d] == 1;
    return ok;
}

bcast get_bcast_type(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d) {
    if (is_full_bcast(src_d, weights_d)) return bcast::full;
    if (!is_per_oc_bcast(src_d, weights_d)) return bcast::unsupported;

    const auto &strides = src_d.blocking_desc().strides;

    // Non-plain dense layouts with per-channel weights are treated as channel
    // blocked layouts, e.g. nChw8c. bcast_supported() validates the block size.
    if (!src_d.is_plain()) {
        return bcast::per_oc_blocked;
    } else if (strides[1] == 1) {
        // Channel is physically contiguous, as in NHWC/NDHWC.
        return bcast::per_oc_n_spatial_c;
    } else if (strides[0] >= strides[1]
            && IMPLICATION(src_d.ndims() >= 3, strides[1] >= strides[2])) {
        // Channel-first dense layout, as in NCHW/NCDHW.
        return bcast::per_oc_n_c_spatial;
    }

    return bcast::unsupported;
}

bool bcast_supported(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, size_t simd_w) {
    const auto bcast_type = get_bcast_type(src_d, weights_d);

    if (bcast_type == bcast::full) return true;
    if (bcast_type == bcast::unsupported) return false;

    if (bcast_type == bcast::per_oc_blocked) {
        // Keep blocked support deliberately narrow: one channel block, on the
        // channel dimension, with block size equal to the active vector width.
        const auto check_block_consistency = [=](const memory_desc_wrapper &d) {
            const auto &bd = d.blocking_desc();
            return bd.inner_nblks == 1
                    && static_cast<size_t>(bd.inner_blks[0]) == simd_w
                    && bd.inner_idxs[0] == 1;
        };

        return check_block_consistency(src_d)
                && check_block_consistency(weights_d);
    }

    const auto &src_strides = src_d.blocking_desc().strides;
    const auto &weights_strides = weights_d.blocking_desc().strides;

    return src_strides[0] >= src_strides[1]
            && IMPLICATION(src_strides[1] > 1, src_strides[1] >= src_strides[2])
            && weights_strides[0] >= weights_strides[1];
}

} // namespace prelu

#define PARAM_OFF(x) offsetof(jit_prelu_forward_kernel_t::call_params_t, x)

template <cpu_isa_t isa>
jit_uni_prelu_forward_kernel_t<isa>::jit_uni_prelu_forward_kernel_t(
        prelu::bcast bcast, data_type_t data_type)
    : jit_prelu_forward_kernel_t(bcast)
    , data_type_(data_type)
    , simd_w_(simd_elems(data_type::f32, isa))
    , simd_bytes_(simd_w_ * types::data_type_size(data_type_)) {}

template <cpu_isa_t isa>
bool jit_uni_prelu_forward_kernel_t<isa>::weights_are_vector_const()
        const noexcept {
    return bcast_ == prelu::bcast::per_oc_blocked;
}

template <cpu_isa_t isa>
bool jit_uni_prelu_forward_kernel_t<isa>::weights_are_scalar_const()
        const noexcept {
    return bcast_ == prelu::bcast::per_oc_n_c_spatial;
}

template <cpu_isa_t isa>
bool jit_uni_prelu_forward_kernel_t<isa>::weights_are_const() const noexcept {
    return weights_are_vector_const() || weights_are_scalar_const();
}

template <cpu_isa_t isa>
size_t jit_uni_prelu_forward_kernel_t<isa>::data_type_size() const noexcept {
    return types::data_type_size(data_type_);
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::load_vector(
        const VReg4S &dst, const XReg &addr) {
    ld1(dst, ptr(addr));
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::load_vector(
        const ZRegS &dst, const XReg &addr) {
    ld1w(dst, P_ALL_ONE / T_z, ptr(addr));
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::load_data_vector(
        const VReg4S &dst, const XReg &addr) {
    if (data_type_ == data_type::f32) {
        ld1(dst, ptr(addr));
    } else if (data_type_ == data_type::f16) {
        ld1(VReg4H(dst.getIdx()), ptr(addr));
        fcvtl(dst, VReg4H(dst.getIdx()));
    } else {
        assert(!"unsupported PReLU data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::load_data_vector(
        const ZRegS &dst, const XReg &addr) {
    if (data_type_ == data_type::f32) {
        ld1w(dst, P_ALL_ONE / T_z, ptr(addr));
    } else if (data_type_ == data_type::f16) {
        ld1h(dst, P_ALL_ONE / T_z, ptr(addr));
        fcvt(dst, P_ALL_ONE / T_m, ZRegH(dst.getIdx()));
    } else {
        assert(!"unsupported PReLU data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::store_vector(
        const XReg &addr, const VReg4S &src) {
    st1(src, ptr(addr));
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::store_vector(
        const XReg &addr, const ZRegS &src) {
    st1w(src, P_ALL_ONE / T_z, ptr(addr));
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::store_data_vector(
        const XReg &addr, const VReg4S &src) {
    if (data_type_ == data_type::f32) {
        st1(src, ptr(addr));
    } else if (data_type_ == data_type::f16) {
        fcvtn(VReg4H(src.getIdx()), src);
        st1(VReg4H(src.getIdx()), ptr(addr));
    } else {
        assert(!"unsupported PReLU data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::store_data_vector(
        const XReg &addr, const ZRegS &src) {
    if (data_type_ == data_type::f32) {
        st1w(src, P_ALL_ONE / T_z, ptr(addr));
    } else if (data_type_ == data_type::f16) {
        fcvt(ZRegH(src.getIdx()), P_ALL_ONE / T_m, src);
        st1h(src, P_ALL_ONE, ptr(addr));
    } else {
        assert(!"unsupported PReLU data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::broadcast_weight(
        const VReg4S &dst, const XReg &addr) {
    if (data_type_ == data_type::f32) {
        ld1r(dst, ptr(addr));
    } else if (data_type_ == data_type::f16) {
        ld1r(VReg4H(dst.getIdx()), ptr(addr));
        fcvtl(dst, VReg4H(dst.getIdx()));
    } else {
        assert(!"unsupported PReLU data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::broadcast_weight(
        const ZRegS &dst, const XReg &addr) {
    if (data_type_ == data_type::f32) {
        ld1rw(dst, P_ALL_ONE / T_z, ptr(addr));
    } else if (data_type_ == data_type::f16) {
        ld1rh(dst, P_ALL_ONE / T_z, ptr(addr));
        fcvt(dst, P_ALL_ONE / T_m, ZRegH(dst.getIdx()));
    } else {
        assert(!"unsupported PReLU data type");
    }
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::compute_vector(
        const VReg4S &src, const VReg4S &weights, const VReg4S &dst) {
    UNUSED(dst);
    const VReg4S v_max(v_max_.getIdx());
    const VReg4S v_min(v_min_.getIdx());
    const VReg4S v_zero(v_zero_.getIdx());

    assert(dst.getIdx() == v_max.getIdx());

    // PReLU: positive part passes through, negative part is multiplied by
    // weights. Algebraically: max(src, 0) + weights * min(src, 0).
    fmax(v_max, src, v_zero);
    fmin(v_min, src, v_zero);
    fmla(v_max, v_min, weights);
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::compute_vector(
        const ZRegS &src, const ZRegS &weights, const ZRegS &dst) {
    UNUSED(dst);
    const ZRegS v_max(v_max_.getIdx());
    const ZRegS v_min(v_min_.getIdx());
    const ZRegS v_zero(v_zero_.getIdx());

    assert(dst.getIdx() == v_max.getIdx());

    // Same math as the ASIMD version, using an all-true SVE predicate.
    mov(v_max, P_ALL_ONE / T_m, src);
    fmax(v_max, P_ALL_ONE / T_m, v_zero);
    mov(v_min, P_ALL_ONE / T_m, src);
    fmin(v_min, P_ALL_ONE / T_m, v_zero);
    fmla(v_max, P_ALL_ONE / T_m, v_min, weights);
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::load_params() {
    ldr(reg_src_, ptr(abi_param1, static_cast<uint32_t>(PARAM_OFF(src))));
    ldr(reg_weights_,
            ptr(abi_param1, static_cast<uint32_t>(PARAM_OFF(weights))));
    ldr(reg_dst_, ptr(abi_param1, static_cast<uint32_t>(PARAM_OFF(dst))));
    ldr(reg_work_,
            ptr(abi_param1,
                    static_cast<uint32_t>(PARAM_OFF(compute_data_size))));
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::prepare_const_registers() {
    uni_clear(TReg(v_zero_.getIdx()));

    // Some broadcast modes can load weights once per kernel call. For full and
    // NHWC-like cases, weights advance with src and are loaded in vector_loop().
    if (weights_are_vector_const())
        load_data_vector(v_weights_, reg_weights_);
    else if (weights_are_scalar_const())
        broadcast_weight(v_weights_, reg_weights_);
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::vector_loop() {
    Label loop, end;

    cmp(reg_work_, simd_w_);
    b(LT, end);

    L(loop);
    load_data_vector(v_src_, reg_src_);
    if (!weights_are_const()) load_data_vector(v_weights_, reg_weights_);
    compute_vector(v_src_, v_weights_, v_max_);
    store_data_vector(reg_dst_, v_max_);

    add_imm(reg_src_, reg_src_, simd_bytes_, X_TMP_0);
    add_imm(reg_dst_, reg_dst_, simd_bytes_, X_TMP_0);
    if (!weights_are_const())
        add_imm(reg_weights_, reg_weights_, simd_bytes_, X_TMP_0);

    sub_imm(reg_work_, reg_work_, simd_w_, X_TMP_0);
    cmp(reg_work_, simd_w_);
    b(GE, loop);

    L(end);
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::scalar_loop() {
    Label loop, end;

    cmp(reg_work_, 0);
    b(LE, end);

    L(loop);
    if (data_type_ == data_type::f32) {
        ldr(s_src_, ptr(reg_src_));
        ldr(s_weights_, ptr(reg_weights_));
    } else if (data_type_ == data_type::f16) {
        ldr(HReg(s_src_.getIdx()), ptr(reg_src_));
        ldr(HReg(s_weights_.getIdx()), ptr(reg_weights_));
        fcvt(s_src_, HReg(s_src_.getIdx()));
        fcvt(s_weights_, HReg(s_weights_.getIdx()));
    } else {
        assert(!"unsupported PReLU data type");
    }
    // Scalar cleanup mirrors the vector formula for plain-layout tails.
    fmax(s_max_, s_src_, s_zero_);
    fmin(s_min_, s_src_, s_zero_);
    fmul(s_min_, s_min_, s_weights_);
    fadd(s_max_, s_max_, s_min_);
    if (data_type_ == data_type::f32) {
        str(s_max_, ptr(reg_dst_));
    } else if (data_type_ == data_type::f16) {
        fcvt(HReg(s_max_.getIdx()), s_max_);
        str(HReg(s_max_.getIdx()), ptr(reg_dst_));
    }

    add_imm(reg_src_, reg_src_, data_type_size(), X_TMP_0);
    add_imm(reg_dst_, reg_dst_, data_type_size(), X_TMP_0);
    if (!weights_are_const())
        add_imm(reg_weights_, reg_weights_, data_type_size(), X_TMP_0);

    sub_imm(reg_work_, reg_work_, 1, X_TMP_0);
    cmp(reg_work_, 0);
    b(GT, loop);

    L(end);
}

template <cpu_isa_t isa>
void jit_uni_prelu_forward_kernel_t<isa>::generate() {
    preamble();
    load_params();
    prepare_const_registers();
    vector_loop();
    scalar_loop();
    postamble();
}

#undef PARAM_OFF

template <cpu_isa_t isa>
status_t jit_uni_prelu_fwd_t<isa>::pd_t::init(engine_t *engine) {
    UNUSED(engine);

    const memory_desc_wrapper src_d {src_md(0)};
    const memory_desc_wrapper weights_d {weights_md(0)};
    const memory_desc_wrapper dst_d {dst_md(0)};
    const auto simd_w = simd_elems(data_type::f32, isa);

    VDISPATCH_PRELU(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_PRELU(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_PRELU(src_d.data_type() == dst_d.data_type(),
            VERBOSE_INCONSISTENT_DT, "src", "dst");
    VDISPATCH_PRELU(utils::everyone_is(src_d.data_type(), weights_d.data_type(),
                            dst_d.data_type()),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_PRELU(
            utils::one_of(src_d.data_type(), data_type::f32, data_type::f16),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_PRELU(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_PRELU(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "src");
    VDISPATCH_PRELU(src_d.is_dense(true), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_PRELU(weights_d.is_dense(true), VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_PRELU(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_PRELU(dst_d == src_d, VERBOSE_INCONSISTENT_MDS, "src", "dst");
    VDISPATCH_PRELU(prelu::bcast_supported(src_d, weights_d, simd_w),
            VERBOSE_UNSUPPORTED_DT_CFG);

    bcast_ = prelu::get_bcast_type(src_d, weights_d);
    data_type_ = src_d.data_type();
    return status::success;
}

template <cpu_isa_t isa>
jit_uni_prelu_fwd_t<isa>::jit_uni_prelu_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa>
jit_uni_prelu_fwd_t<isa>::~jit_uni_prelu_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_prelu_fwd_t<isa>::init(engine_t *engine) {
    UNUSED(engine);

    CHECK(safe_ptr_assign(kernel_,
            new jit_uni_prelu_forward_kernel_t<isa>(
                    pd()->bcast_, pd()->data_type_)));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_prelu_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    using byte = unsigned char;

    const byte *const src = CTX_IN_MEM(const byte *, DNNL_ARG_SRC);
    const byte *const weights = CTX_IN_MEM(const byte *, DNNL_ARG_WEIGHTS);
    byte *const dst = CTX_OUT_MEM(byte *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d {pd()->src_md(0)};
    const auto kernel = kernel_.get();
    const auto bcast = kernel->get_bcast();
    const size_t simd_w = kernel->simd_w();
    const size_t data_type_size = types::data_type_size(pd()->data_type_);

    const dim_t ndims = src_d.ndims();
    const dim_t MB = pd()->N();
    const dim_t C = pd()->C();
    const dim_t D = pd()->D();
    const dim_t H = pd()->H();
    const dim_t W = pd()->W();
    const dim_t SP = D * H * W;

    if (bcast == prelu::bcast::full) {
        const dim_t nelems = src_d.nelems(true);
        const dim_t work_chunks = utils::div_up(nelems, (dim_t)simd_w);

        // Full weights are laid out exactly like src/dst, so the tensor can be
        // treated as one flat dense array split into SIMD-sized chunks.
        parallel(0, [=](const int ithr, const int nthr) {
            dim_t start = 0, end = 0;
            balance211(work_chunks, nthr, ithr, start, end);
            start = nstl::min(nelems, start * (dim_t)simd_w);
            end = nstl::min(nelems, end * (dim_t)simd_w);
            if (start >= end) return;

            jit_prelu_forward_kernel_t::call_params_t params;
            params.compute_data_size = end - start;
            params.src = src + start * data_type_size;
            params.weights = weights + start * data_type_size;
            params.dst = dst + start * data_type_size;
            (*kernel)(&params);
        });

        return status::success;
    }

    const dim_t nelems_single_mb
            = utils::array_product(src_d.padded_dims() + 1, ndims - 1);

    if (bcast == prelu::bcast::per_oc_n_spatial_c) {
        // NHWC-like case. For each minibatch/spatial position, process the
        // contiguous channel dimension and advance weights alongside channels.
        parallel_nd(MB, SP, [=](dim_t mb, dim_t sp) {
            const dim_t offset = mb * nelems_single_mb + sp * C;

            jit_prelu_forward_kernel_t::call_params_t params;
            params.compute_data_size = C;
            params.src = src + offset * data_type_size;
            params.weights = weights;
            params.dst = dst + offset * data_type_size;
            (*kernel)(&params);
        });
    } else if (bcast == prelu::bcast::per_oc_n_c_spatial) {
        // NCHW-like case. Each kernel call handles one channel's spatial range,
        // so a single scalar weight can be broadcast and reused.
        parallel_nd(MB, C, [=](dim_t mb, dim_t c) {
            const dim_t offset = mb * nelems_single_mb + c * SP;

            jit_prelu_forward_kernel_t::call_params_t params;
            params.compute_data_size = SP;
            params.src = src + offset * data_type_size;
            params.weights = weights + c * data_type_size;
            params.dst = dst + offset * data_type_size;
            (*kernel)(&params);
        });
    } else if (bcast == prelu::bcast::per_oc_blocked) {
        const dim_t C_blocks = utils::div_up(C, (dim_t)simd_w);

        // Blocked channel layout. One vector of channel weights is loaded for a
        // block and reused across every spatial element in that block.
        parallel_nd(MB, C_blocks, [=](dim_t mb, dim_t c_blk) {
            const dim_t offset = mb * nelems_single_mb + c_blk * SP * simd_w;

            jit_prelu_forward_kernel_t::call_params_t params;
            params.compute_data_size = SP * simd_w;
            params.src = src + offset * data_type_size;
            params.weights = weights + c_blk * simd_w * data_type_size;
            params.dst = dst + offset * data_type_size;
            (*kernel)(&params);
        });
    } else {
        assert(!"unsupported PReLU broadcast type");
        return status::runtime_error;
    }

    return status::success;
}

template class jit_uni_prelu_forward_kernel_t<asimd>;
template class jit_uni_prelu_forward_kernel_t<sve>;
template struct jit_uni_prelu_fwd_t<asimd>;
template struct jit_uni_prelu_fwd_t<sve>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
