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

#ifndef CPU_AARCH64_PRELU_JIT_UNI_PRELU_FORWARD_HPP
#define CPU_AARCH64_PRELU_JIT_UNI_PRELU_FORWARD_HPP

#include <memory>

#include "common/primitive.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/cpu_prelu_pd.hpp"

namespace dnnl {
namespace impl {

struct memory_desc_wrapper;

namespace cpu {
namespace aarch64 {

namespace prelu {

// The JIT kernel is shared across several PReLU broadcast patterns. The
// primitive descriptor classifies the memory descriptors once, then execute()
// uses this value to choose how to pass pointers and work sizes to the kernel.
enum class bcast {
    full,
    per_oc_blocked,
    per_oc_n_spatial_c,
    per_oc_n_c_spatial,
    unsupported
};

bcast get_bcast_type(
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d);
bool bcast_supported(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, size_t simd_w);

} // namespace prelu

class jit_prelu_forward_kernel_t : public jit_generator_t {
public:
    // Per-call kernel state. The C++ primitive wrapper slices the tensor into
    // contiguous chunks and passes each chunk through this compact ABI.
    struct call_params_t {
        const void *src = nullptr;
        const void *weights = nullptr;
        void *dst = nullptr;
        size_t compute_data_size = 0;
    };

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_prelu_forward_kernel_t)

    void operator()(call_params_t *params) {
        jit_generator_t::operator()(params);
    }

    virtual size_t simd_w() const noexcept = 0;
    virtual prelu::bcast get_bcast() const noexcept = 0;

protected:
    explicit jit_prelu_forward_kernel_t(prelu::bcast bcast) : bcast_(bcast) {}

    const prelu::bcast bcast_;
};

template <cpu_isa_t isa>
class jit_uni_prelu_forward_kernel_t : public jit_prelu_forward_kernel_t {
public:
    explicit jit_uni_prelu_forward_kernel_t(
            prelu::bcast bcast, data_type_t data_type);

    size_t simd_w() const noexcept override { return simd_w_; }
    prelu::bcast get_bcast() const noexcept override { return bcast_; }

private:
    using TReg = typename cpu_isa_traits<isa>::TReg;
    using TRegS = typename cpu_isa_traits<isa>::TRegS;

    // Code-generation steps for the emitted function body.
    void generate() override;
    void load_params();
    void prepare_const_registers();
    void vector_loop();
    void scalar_loop();

    // ISA-specific helpers. The overloads let the common kernel body emit
    // either ASIMD or SVE instructions from the same template.
    void load_vector(
            const Xbyak_aarch64::VReg4S &dst, const Xbyak_aarch64::XReg &addr);
    void load_vector(
            const Xbyak_aarch64::ZRegS &dst, const Xbyak_aarch64::XReg &addr);
    void store_vector(
            const Xbyak_aarch64::XReg &addr, const Xbyak_aarch64::VReg4S &src);
    void store_vector(
            const Xbyak_aarch64::XReg &addr, const Xbyak_aarch64::ZRegS &src);
    void broadcast_weight(
            const Xbyak_aarch64::VReg4S &dst, const Xbyak_aarch64::XReg &addr);
    void broadcast_weight(
            const Xbyak_aarch64::ZRegS &dst, const Xbyak_aarch64::XReg &addr);
    void compute_vector(const Xbyak_aarch64::VReg4S &src,
            const Xbyak_aarch64::VReg4S &weights,
            const Xbyak_aarch64::VReg4S &dst);
    void compute_vector(const Xbyak_aarch64::ZRegS &src,
            const Xbyak_aarch64::ZRegS &weights,
            const Xbyak_aarch64::ZRegS &dst);
    void load_data_vector(
            const Xbyak_aarch64::VReg4S &dst, const Xbyak_aarch64::XReg &addr);
    void load_data_vector(
            const Xbyak_aarch64::ZRegS &dst, const Xbyak_aarch64::XReg &addr);
    void store_data_vector(
            const Xbyak_aarch64::XReg &addr, const Xbyak_aarch64::VReg4S &src);
    void store_data_vector(
            const Xbyak_aarch64::XReg &addr, const Xbyak_aarch64::ZRegS &src);

    bool weights_are_const() const noexcept;
    bool weights_are_vector_const() const noexcept;
    bool weights_are_scalar_const() const noexcept;
    size_t data_type_size() const noexcept;

    const data_type_t data_type_;
    const size_t simd_w_;
    const size_t simd_bytes_;

    // General-purpose registers used by the generated kernel.
    const Xbyak_aarch64::XReg reg_src_ = x8;
    const Xbyak_aarch64::XReg reg_weights_ = x9;
    const Xbyak_aarch64::XReg reg_dst_ = x10;
    const Xbyak_aarch64::XReg reg_work_ = x11;

    // Vector/SVE registers. TRegS maps to VReg4S for ASIMD and ZRegS for SVE.
    const TRegS v_src_ = TRegS(0);
    const TRegS v_max_ = TRegS(1);
    const TRegS v_min_ = TRegS(2);
    const TRegS v_weights_ = TRegS(3);
    const TRegS v_zero_ = TRegS(4);

    // Scalar registers used for the cleanup loop after full vectors are done.
    const Xbyak_aarch64::SReg s_src_ = s0;
    const Xbyak_aarch64::SReg s_max_ = s1;
    const Xbyak_aarch64::SReg s_min_ = s2;
    const Xbyak_aarch64::SReg s_weights_ = s3;
    const Xbyak_aarch64::SReg s_zero_ = s4;
};

template <cpu_isa_t isa>
struct jit_uni_prelu_fwd_t : public primitive_t {
    struct pd_t : public cpu_prelu_fwd_pd_t {
        using cpu_prelu_fwd_pd_t::cpu_prelu_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""), jit_uni_prelu_fwd_t);

        status_t init(engine_t *engine);

        // Broadcast kind selected during primitive descriptor initialization.
        // execute() and the JIT kernel both use this to choose pointer movement.
        prelu::bcast bcast_ = prelu::bcast::unsupported;
        data_type_t data_type_ = data_type::undef;
    };

    explicit jit_uni_prelu_fwd_t(const pd_t *apd);
    ~jit_uni_prelu_fwd_t() override;

    status_t init(engine_t *engine) override;
    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    std::unique_ptr<jit_prelu_forward_kernel_t> kernel_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
