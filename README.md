# cosmology

📖 This is an archive for some functions to process cosmology problems.

## Structure

## Functions
### data

### growth
- `growth.D(z, m0, Lambda0, r0) -> z.shape()` 
  Calculate the linear growth factor $D(z)$ with cosmological parameters $\Omega_{m,0},\Omega_{\Lambda,0},\Omega_{r,0} $, where $m,\Lambda, r$ refer to matter, cosmological constant (dark energy)  and radiation. Especially, if  $\Omega_{m,0}+\Omega_{\Lambda,0}+\Omega_{r,0} \neq1$, it will calculate the curvature $\Omega_{k,0}$ which growths as $(1+z)^2$.
  #### parameters:
  - `z`: _ArrayLike_  || redshift
  - `m0, Lambda0, r0`" _float_ || $\Omega_{m,0},\Omega_{\Lambda,0},\Omega_{r,0} $


### power_spectrum