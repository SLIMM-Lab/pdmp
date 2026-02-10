# CompositeTransformation Visual Explanation

## Concept Overview

```
Input (unbounded):          ξ = [ξ₀, ξ₁, ξ₂, ξ₃, ξ₄]
                                  │   │   │   │   │
                                  └───┘   └───────┘
                                    │         │
                             Transform 1  Transform 2
                               (Sigmoid)  (Exponential)
                                    │         │
                                  ┌─┴─┐   ┌───┴────┐
                                  │   │   │   │    │
Output (constrained):         x = [x₀, x₁, x₂, x₃, x₄]
                                  
                            x₀,x₁ ∈ [0,1]   x₂,x₃,x₄ > 0
```

## Block-Diagonal Jacobian Structure

```
     ┌                           ┐
     │  J₁₁  J₁₂   0    0    0  │
     │  J₂₁  J₂₂   0    0    0  │
J =  │   0    0   J₃₃  J₃₄  J₃₅ │
     │   0    0   J₄₃  J₄₄  J₄₅ │
     │   0    0   J₅₃  J₅₄  J₅₅ │
     └                           ┘
     
     └──Block 1──┘ └────Block 2────┘
      (Sigmoid)      (Exponential)
```

**Key Property**: `log |det(J)| = log |det(J_block1)| + log |det(J_block2)|`

## Comparison: Composite vs Chained

### Composite (What We Implemented)

```
Input:  ξ = [ξ₀, ξ₁, ξ₂, ξ₃, ξ₄]
             │   │   │   │   │
             │   │   └───┴───┴── Transform B
             │   │       
             └───┴─────────────── Transform A
             
Output: x = [A(ξ₀,ξ₁), B(ξ₂,ξ₃,ξ₄)]
```

**Properties**:
- Different transforms for different dimensions
- Block-diagonal Jacobian
- O(n) complexity for many operations

### Chained (Alternative)

```
Input:  ξ = [ξ₀, ξ₁, ξ₂, ξ₃, ξ₄]
             │   │   │   │   │
             └───┴───┴───┴───┴── Transform A
                     │
                     └─────────── Transform B
             
Output: x = B(A(ξ))
```

**Properties**:
- Same transform sequence for all dimensions
- Full Jacobian (generally)
- Chain rule for derivatives

## Use Case Examples

### Example 1: Statistical Model

```
Parameters:
├── μ (mean)         ─────→ ℝ (unbounded)      ─→ Identity/Affine
├── σ² (variance)    ─────→ (0, ∞) (positive) ─→ Exponential
└── p (probability)  ─────→ [0, 1] (bounded)  ─→ Sigmoid

Composite:
  ξ = [ξ_μ, ξ_σ², ξ_p]
       │     │     │
       │     exp   sigmoid([0,1])
       │     │     │
  x = [μ,    σ²,   p]
```

### Example 2: Physics Simulation

```
Variables:
├── Temperature (T)      ─→ (0, ∞)     ─→ Exponential
├── Concentration (c)    ─→ [0, 1]     ─→ Sigmoid
├── Position (x, y)      ─→ ℝ²         ─→ Affine
└── Velocity (vx, vy)    ─→ ℝ²         ─→ Affine

Composite:
  ξ = [ξ_T, ξ_c, ξ_x, ξ_y, ξ_vx, ξ_vy]
       │     │    │    │    │     │
       exp   sig  └────┴────┴─────┴─── Affine transform
       │     │           │
  x = [T,    c,     x, y, vx, vy]
```

### Example 3: Hierarchical Model

```
Level 1 (Hyperparameters):  [α, β]  ─→ (0, ∞)  ─→ Exponential
Level 2 (Parameters):       [θ₁, θ₂, θ₃] ─→ [0, 1] ─→ Sigmoid

Joint:
  ξ = [ξ_α, ξ_β, ξ_θ₁, ξ_θ₂, ξ_θ₃]
       │    │     │     │     │
       └────┘     └─────┴─────┘
         exp         sigmoid
         │           │
  x = [α, β,  θ₁, θ₂, θ₃]
```

## Transformation Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│         CompositeTransformation                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input: ξ ∈ ℝᵈ (unconstrained)                          │
│         │                                                │
│         ├──→ Indices [0,1] ──→ Transform₁ ──┐          │
│         │                                     │          │
│         ├──→ Indices [2,3,4] ─→ Transform₂ ─┼──→ x     │
│         │                                     │          │
│         └──→ Indices [5] ────→ Transform₃ ──┘          │
│                                                          │
│  Output: x = [x₀, x₁, x₂, x₃, x₄, x₅]                  │
│          Each subset satisfies its constraint           │
└─────────────────────────────────────────────────────────┘
```

## Automatic Index Inference (JointDistribution)

```
┌───────────────────────────────────┐
│      JointDistribution             │
│                                    │
│  dist₁ (dim=2)  ──→ indices [0,1] │
│  dist₂ (dim=3)  ──→ indices [2,3,4]│
│  dist₃ (dim=1)  ──→ indices [5]   │
└───────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│   CompositeTransformation          │
│                                    │
│  Automatically applies:            │
│    trans₁ to dims [0,1]           │
│    trans₂ to dims [2,3,4]         │
│    trans₃ to dim [5]              │
└───────────────────────────────────┘
```

## Gradient Computation

```
For log p(ξ) where x = T(ξ):

log p(ξ) = log p_X(T(ξ)) + log |det J_T(ξ)|
           └─────┬─────┘   └──────┬──────┘
         Base density    Change of variables

∇ log p(ξ) = J_T(ξ)ᵀ ∇ log p_X(x) + ∇ log |det J_T(ξ)|
             └──────┬────────┘      └────────┬─────────┘
          Transformed gradient       Jacobian adjustment

For composite: ∇ log |det J_T(ξ)| = ∑ᵢ ∇ log |det J_Tᵢ(ξᵢ)|
                                     Block-wise sum
```

## Memory and Complexity

### Storage

```
Full Jacobian:     O(d²)
Block-diagonal:    O(d₁² + d₂² + ... + dₖ²)
                   
For equal blocks:  O(k·(d/k)²) = O(d²/k)
                   k-fold reduction!
```

### Computation

```
Operation              Full Matrix    Block-Diagonal
────────────────────────────────────────────────────
Matrix multiplication    O(d³)          O(d₁³ + d₂³ + ...)
Determinant             O(d³)          O(d₁³ + d₂³ + ...)
Inverse                 O(d³)          O(d₁³ + d₂³ + ...)

For equal blocks:      O(d³)          O(k·(d/k)³) = O(d³/k²)
```

## Decision Tree: When to Use Composite

```
                    Start
                      │
              ┌───────┴────────┐
              │ Different       │
              │ constraints     │
              │ for different   │
              │ variables?      │
              └───────┬─────────┘
                  Yes │   No
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
    ┌───────────────┐   ┌──────────────┐
    │ Use COMPOSITE │   │ Use single   │
    │ Transformation│   │ transform or │
    │               │   │ chained      │
    └───────────────┘   └──────────────┘
            │
            │
    ┌───────┴────────┐
    │ Have Joint     │
    │ Distribution?  │
    └───────┬────────┘
        Yes │   No
      ┌─────┴─────┐
      │           │
      ▼           ▼
┌──────────┐  ┌────────────┐
│ Omit     │  │ Specify    │
│ indices  │  │ indices    │
│ (auto)   │  │ explicitly │
└──────────┘  └────────────┘
```

