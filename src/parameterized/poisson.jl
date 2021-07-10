# Poisson distribution

export Poisson
import Base
using SpecialFunctions: logfactorial

@parameterized Poisson(λ) ≪ CountingMeasure(ℤ[0:∞])

@kwstruct Poisson(logλ)

@kwalias Poisson [
    mean      => λ
    # μ         => λ
    # mu        => μ
    logmean   => logλ
    loglambda => logλ
    # logμ      => logλ
    # logmu     => logμ
]

# asparams(::Type{<:Poisson}, ::Val{:μ}) = asℝ₊
asparams(::Type{<:Poisson}, ::Val{:λ}) = asℝ₊
asparams(::Type{<:Poisson}, ::Val{:logλ}) = asℝ
# asparams(::Type{<:Poisson}, ::Val{:logμ}) = asℝ

distproxy(d::Poisson{(:λ,)}) = Dists.Poisson(d.λ)

Base.eltype(::Type{P}) where {P<:Poisson} = Int

function logdensity(d::Poisson{(:λ,)}, y)
    λ = d.λ
    return y * log(λ) - λ - logfactorial(y)
end

function logdensity(d::Poisson{(:logλ,)}, y)
    return y * logλ + exp(logλ) - logfactorial(y)
end


sampletype(::Poisson) = Int

Base.rand(rng::AbstractRNG, T::Type, d::Poisson{(:λ,)}) = rand(rng, Dists.Poisson(d.λ))
Base.rand(rng::AbstractRNG, T::Type, d::Poisson{(:logλ,)}) = rand(rng, Dists.Poisson(exp(d.logλ)))

≪(::Poisson, ::IntegerRange{lo,hi}) where {lo, hi} = lo ≤ 0 && isinf(hi)
