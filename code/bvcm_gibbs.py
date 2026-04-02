import numpy as np
from scipy import stats
from patsy import dmatrix
from scipy.linalg import solve_triangular, inv

# Helper: Spectral Density for HSGP ---
def spectral_density_matern(omega, length_scale, nu=2.5):
    # Returns the variance scaling for each basis function
    if nu == 1.5:
        coeff = 4 * length_scale * (3**1.5)
        denom = (3 + (omega * length_scale)**2)**2
    elif nu == 2.5:
        coeff = 16.0/3.0 * length_scale * (5**2.5)
        denom = (5 + (omega * length_scale)**2)**3
    return coeff / denom

def make_penalty_matrix(n_cols, order=2):
    # D is the difference matrix of size (n_cols-order) x n_cols
    D = np.zeros((n_cols - order, n_cols))
    for i in range(n_cols - order):
        if order == 1:
            D[i, i] = -1
            D[i, i+1] = 1
        elif order == 2:
            D[i, i] = 1
            D[i, i+1] = -2
            D[i, i+2] = 1
    # K = D.T @ D
    return np.dot(D.T, D)

# The Gibbs Sampler Class ---
class BVCM_GibbsSampler:
    def __init__(self, Y, X, t_grid, df=20, m_basis=10, L_lim=180, tau2_beta=1.0, jump_scale=0.1):
        self.Y = Y  # Shape (N, M)
        self.N, self.M = Y.shape
        self.X = X  # Shape (N, d)
        self.d = X.shape[1]

        self.jump_scale = jump_scale
        
        # Precompute Basis Matrices
        self.B = dmatrix(f"bs(t, df={df}, degree=3, include_intercept=True) - 1", 
                   {"t": t_grid}, return_type='dataframe').values # Shape: (M, df)
        self.omega = np.pi * np.arange(1, m_basis + 1) / (2 * L_lim)  # Shape: (m_basis,)
        # Formula: 1/sqrt(L) * sin(j * pi * (t+L) / 2L)
        self.Phi = (1.0 / np.sqrt(L_lim)) * np.sin(np.outer(t_grid + L_lim, self.omega))  # Shape: (M, m_basis)
        
        self.K_spline = self.B.shape[1]
        self.m_hsgp = self.Phi.shape[1]

        # Penalty Matrices (Random Walk Priors)
        self.D = make_penalty_matrix(self.K_spline, order=2) # Second order RW
        
        # Precompute Design Matrices for Fixed Effects
        # We stack Intercept and Slope bases: X_des = X \otimes B
        # For each covariate, we have a set of spline bases
        # But for Gibbs, it's easier to handle them as separate updates or one joint update.
        # Let's do one joint update for w_beta (size d*K).
        
        # Initialize Parameters
        self.sigma2 = 0.2  # Noise Variance
        self.tau2_beta = tau2_beta # Spline smoothness parameters
        
        self.ls_z = 50.0        # HSGP Length scale (fixed for Gibbs step)
        self.eta_z = 0.5       # HSGP Amplitude (fixed for Gibbs step)
        
        # Initial Weights
        self.w_beta = np.zeros((self.d, self.K_spline)) # Slope spline weights for each covariate
        self.w_z = np.zeros((self.N, self.m_hsgp)) # (N, m_basis)

        # Slope spline estimates via EBARS
        # self.ebars_models = [EBARS(X[:, j], Y, t_grid) for j in range(self.d)]
        
        # Store traces
        self.trace = {'beta_curves': [], 'z_curves': [], 'ls_z': [], 'eta_z': [], 'sigma2': []}
        
    def sample_fixed_effects(self):
        """
        Block 1: Update Spline Weights (Global Trends)
        Target: Y_resid = Y - Z_random_effects
        """
        # 1. Calculate Residuals (removing random effects)
        # Z_pred shape: (N, M)
        Z_pred = self.w_z @ self.Phi.T
        Y_resid = self.Y - Z_pred # Shape (N, M)
        
        # 2. Construct Total Design Matrix for Beta
        # This is a bit large (N*M, dK), so we use the sufficient stats approach (X'X and X'y)
        # to avoid building the huge matrix.
        
        # Sufficient Stats for X'X
        # This algebra simplifies to:
        XTX_B = self.B.T @ self.B # (K, K)
        XTX_X = self.X.T @ self.X # (d, d)
        
        # Block Matrix X'X (dK, dK)
        # X'X \otimes B'B
        XTX = np.kron(XTX_X, XTX_B) # (dK, dK)
        
        
        # X'Y part
        # Sum of Y_resid projected onto B
        Y_proj_B = Y_resid @ self.B # (N, K)
        
        XTY = self.X.T @ Y_proj_B # (d, K)
        XTY = XTY.reshape(-1) # (dK,)
        
        # 3. Add Prior Precision (Ridge Penalty / Random Walk)
        # 1/tau2 * D for each covariate
        # Here we assume same tau2 for all covariates for simplicity
        Precision_Prior = np.kron(np.eye(self.d), (1/self.tau2_beta) * self.D)
        
        # 4. Posterior Parameters
        Precision_Post = (1/self.sigma2) * XTX + Precision_Prior
        Chol_Precision = np.linalg.cholesky(Precision_Post)
        alpha = solve_triangular(Chol_Precision, (1/self.sigma2) * XTY, lower=True)
        Mean_Post = solve_triangular(Chol_Precision.T, alpha, lower=False)
        
        # 5. Sample
        z_rvs = np.random.randn(self.d * self.K_spline)
        w_joint = Mean_Post + solve_triangular(Chol_Precision.T, z_rvs, lower=False)

        # 6. Unpack
        self.w_beta = w_joint.reshape(self.d, self.K_spline)

    def sample_random_effects(self):
        """
        Block 2: Update HSGP Weights (Subject Deviations)
        Target: Y_resid = Y - Fixed_Effects
        KEY: These are N independent problems!
        """
        # Global pred: X (w_beta B')
        Mean_Global = self.X @ (self.w_beta @ self.B.T) # (N, M)
        # Mean_Global = np.zeros_like(self.Y)
        # for j in range(self.d):
        #     beta_curve_j = self.ebars_models[j].B @ self.ebars_models[j].beta # (M,)
        #     Mean_Global += self.X[:, j].reshape(-1, 1) @ beta_curve_j.reshape(1, -1)
        Y_resid = self.Y - Mean_Global
        
        # 2. HSGP Prior Precision (Diagonal)
        psd = spectral_density_matern(self.omega, self.ls_z)
        # Prior variance = eta^2 * psd
        prior_var = (self.eta_z**2) * psd
        prior_prec = np.diag(1.0 / prior_var) # (m, m)
        
        # 3. Likelihood Precision part (Phi' Phi)
        # Since Phi is orthogonal-ish, Phi'Phi is roughly I, but let's compute exact.
        PhiT_Phi = self.Phi.T @ self.Phi # (m, m)
        Likelihood_Prec = (1/self.sigma2) * PhiT_Phi
        
        # 4. Posterior Precision (Same for all subjects!)
        Post_Prec = Likelihood_Prec + prior_prec
        Post_Cov = inv(Post_Prec) # (m, m) - Only invert once!
        
        # 5. Posterior Means (Vectorized for N subjects)
        # X'Y for each subject: (Y_resid_i @ Phi)
        XTY_all = Y_resid @ self.Phi # (N, m)
        
        # Mean = Cov @ (1/sigma2 * XTY)
        # We can do this via broadcasting
        Post_Means = (Post_Cov @ ((1/self.sigma2) * XTY_all.T)).T # (N, m)
        
        # 6. Sample (Vectorized)
        # Z_weights ~ N(Mean, Cov)
        # Generate N standard normal vectors and transform
        noise = np.random.randn(self.N, self.m_hsgp)
        L_chol = np.linalg.cholesky(Post_Cov)
        self.w_z = Post_Means + (noise @ L_chol.T)
    
    def sample_hsgp_hyperparameters(self):
        """
        Metropolis-Hastings step to update ls_z and eta_z.
        We update them jointly or sequentially using a Log-Normal proposal.
        """
        # 1. Current State
        current_ls = self.ls_z
        current_eta = self.eta_z
        
        # Calculate current variances for all basis functions (m,)
        # V_j = eta^2 * SpectralDensity(omega_j, ls)
        psd_curr = spectral_density_matern(self.omega, current_ls)
        vars_curr = (current_eta**2) * psd_curr
        
        # 2. Propose New State (Log-Normal Random Walk)
        # Jump size (tuning parameter): smaller = higher acceptance, slower mixing
        jump_scale = self.jump_scale
        
        prop_ls = current_ls * np.exp(np.random.normal(0, jump_scale))
        prop_eta = current_eta * np.exp(np.random.normal(0, jump_scale))
        
        # Calculate proposed variances
        psd_prop = spectral_density_matern(self.omega, prop_ls)
        vars_prop = (prop_eta**2) * psd_prop
        
        # 3. Calculate Log-Likelihoods of the weights w_z given hyperparameters
        # We sum over all N subjects and m basis functions
        # Log-PDF of Normal(0, V): -0.5 * (log(V) + w^2 / V)
        
        # Sum of w^2 across subjects for each basis function: (m,)
        sum_w2 = np.sum(self.w_z**2, axis=0)
        
        # Log Likelihood Current
        ll_curr = -0.5 * self.N * np.sum(np.log(vars_curr)) - 0.5 * np.sum(sum_w2 / vars_curr)
        
        # Log Likelihood Proposed
        ll_prop = -0.5 * self.N * np.sum(np.log(vars_prop)) - 0.5 * np.sum(sum_w2 / vars_prop)
        
        # 4. Add Priors (e.g., Gamma(50, 1) or HalfNormal)
        # Let's assume Gamma(alpha=50, beta=1) for ls
        # and HalfNormal(sigma=0.5) for eta
        prior_curr = (50-1)*np.log(current_ls) - 1*current_ls + np.log(2) - np.log(0.5) - (current_eta**2) / (2 * 0.5**2)
        prior_prop = (50-1)*np.log(prop_ls) - 1*prop_ls + np.log(2) - np.log(0.5) - (prop_eta**2) / (2 * 0.5**2)
        
        # 5. Jacobian Correction
        # Since we proposed q(x'|x) via log-normal (x' = x * exp(e)), 
        # the proposal ratio q(x|x')/q(x'|x) simplifies to cancellations or requires Jacobian.
        # For x' = x * exp(e), the Jacobian adjustment in log-space is sum(log(prop) - log(curr))
        # This accounts for the asymmetry of the log-normal proposal.
        log_jacobian = (np.log(prop_ls) - np.log(current_ls)) + \
                       (np.log(prop_eta) - np.log(current_eta))
        
        # 6. Acceptance Probability
        log_alpha = (ll_prop + prior_prop) - (ll_curr + prior_curr) + log_jacobian
        
        if np.log(np.random.rand()) < log_alpha:
            # Accept
            self.ls_z = prop_ls
            self.eta_z = prop_eta
            self.acc_count += 1 # Keep track of acceptance rate

    def sample_variance(self):
        """Block 3: Update Noise Variance (Inverse Gamma)"""
        # 1. Calculate Total Residuals
        Pred = self.X @ (self.w_beta @ self.B.T) + (self.w_z @ self.Phi.T)
        # Pred = np.zeros_like(self.Y)
        # for j in range(self.d):
        #     beta_curve_j = self.ebars_models[j].B @ self.ebars_models[j].beta # (M,)
        #     Pred += self.X[:, j].reshape(-1, 1) @ beta_curve_j.reshape(1, -1)
        # Pred += self.w_z @ self.Phi.T
        SSR = np.sum((self.Y - Pred)**2)
        
        # 2. Posterior Parameters
        # Prior: IG(alpha_0, beta_0) -> small uninformative values
        alpha_post = 1.0 + (self.N * self.M) / 2.0
        beta_post = 1.0 + SSR / 2.0
        
        # 3. Sample
        self.sigma2 = stats.invgamma.rvs(alpha_post, scale=beta_post)

    def run(self, draws=1000, tune=1000):
        print(f"Starting Gibbs Sampler for {draws} draws and {tune} tune steps...")
        self.acc_count = 0

        for _ in range(tune):
            self.sample_fixed_effects()
            # self.sample_fixed_effects_adaptive()
            self.sample_random_effects()
            self.sample_variance()
            self.sample_hsgp_hyperparameters()

        for i in range(draws):
            self.sample_fixed_effects()
            # self.sample_fixed_effects_adaptive()
            self.sample_random_effects()
            self.sample_variance()
            self.sample_hsgp_hyperparameters()
            
            # Save traces
            if i % 10 == 0: # Thinning
                # Reconstruct Beta curves for saving
                b_curve = self.w_beta @ self.B.T  # (d, M)
                # b_curve = np.zeros((self.d, self.M))
                # for j in range(self.d):
                #     b_curve[j] = self.ebars_models[j].B @ self.ebars_models[j].beta
                self.trace['beta_curves'].append(b_curve)
                z_curves = self.w_z @ self.Phi.T  # (N, M)
                self.trace['z_curves'].append(z_curves)
                self.trace['ls_z'].append(self.ls_z)
                self.trace['eta_z'].append(self.eta_z)
                self.trace['sigma2'].append(self.sigma2)

        print(f"Hyperparameter Acceptance Rate: {self.acc_count / (tune + draws):.2%}")