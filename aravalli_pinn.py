# %% [markdown]
# # Aravalli Hills Demolition: Enhanced Dual Pollutant Simulation
# ## 60-Day Analysis | 4000 Epochs Training | Complete Visualization Suite

# %% [code]
# First, install required packages
!pip install torch numpy matplotlib seaborn scipy

# %% [code]
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"GPU available: {torch.cuda.is_available()}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# %% [markdown]
# ## Enhanced Dual Pollutant PINN Class

# %% [code]
class EnhancedDualPollutantPINN(nn.Module):
    """
    Enhanced PINN for PM2.5 and PM10 simulation with 60-day analysis
    """
    def __init__(self, layers=[3, 128, 128, 128, 128, 2]):  # Deeper network
        super().__init__()
        self.layers = layers
        self.activation = nn.Tanh()

        # Build enhanced neural network
        self.linears = nn.ModuleList()
        for i in range(len(layers)-1):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.linears:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y, t):
        """Forward pass: returns [PM2.5, PM10] concentrations"""
        X = torch.cat([x, y, t], dim=1)

        for i in range(len(self.linears)-1):
            X = self.activation(self.linears[i](X))

        X = self.linears[-1](X)
        return X  # Shape: [batch_size, 2]

# %% [markdown]
# ## Enhanced 60-Day Simulation Class

# %% [code]
class EnhancedAravalliModel:
    def __init__(self, device='cpu'):
        self.device = device

        # Delhi geographical boundaries (approximate)
        self.x_min, self.x_max = 0, 50  # km (East-West)
        self.y_min, self.y_max = 0, 40  # km (North-South)
        self.t_min, self.t_max = 0, 60  # EXTENDED TO 60 DAYS

        # Enhanced pollutant-specific parameters for 60-day simulation
        self.pollutants = {
            'PM2.5': {
                'D': 0.5,           # Diffusion coefficient (km²/day)
                'S_max': 120.0,     # Increased peak source for 60-day simulation
                'decay_rate': 0.08, # Slower decay for longer simulation
                'background': 20.0,
                'color': '#1f77b4',  # Blue
                'label': 'PM2.5',
                'safe_level': 25,   # WHO guideline
                'danger_level': 60  # Severe level
            },
            'PM10': {
                'D': 0.8,           # Faster diffusion
                'S_max': 180.0,     # Higher emissions
                'decay_rate': 0.12, # Faster settling
                'background': 30.0,
                'color': '#ff7f0e',  # Orange
                'label': 'PM10',
                'safe_level': 50,   # WHO guideline
                'danger_level': 100 # Severe level
            }
        }

        # Locations
        self.aravalli_x, self.aravalli_y = 35, 15
        self.delhi_x, self.delhi_y = 25, 20

        # Storage for loss tracking
        self.total_losses = []
        self.physics_losses = []
        self.boundary_losses = []

    def source_term(self, x, y, t, pollutant='PM2.5'):
        """Enhanced source term for 60-day simulation"""
        params = self.pollutants[pollutant]

        # Distance from Aravalli
        distance = torch.sqrt((x - self.aravalli_x)**2 + (y - self.aravalli_y)**2)

        # Extended Gaussian time profile (peak at day 15, wider spread)
        time_factor = torch.exp(-(t - 15)**2 / 120.0)  # Wider for 60 days

        # Spatial spread
        spatial_factor = torch.exp(-distance**2 / 25.0)  # Slightly wider spread

        # Wind modulation with seasonal variation
        wind_factor = 1.0 + 0.4 * torch.sin(2 * np.pi * t / 15)  # 15-day cycle

        return params['S_max'] * time_factor * spatial_factor * wind_factor

    def wind_field(self, x, y, t):
        """Wind field with seasonal variations"""
        u_base = 2.0 + 0.5 * torch.sin(2 * np.pi * t / 30)  # Monthly variation
        v_base = -2.0 + 0.3 * torch.cos(2 * np.pi * t / 30)

        u_var = 0.6 * torch.sin(2 * np.pi * x / self.x_max) * torch.cos(2 * np.pi * t / 10)
        v_var = 0.6 * torch.cos(2 * np.pi * y / self.y_max) * torch.sin(2 * np.pi * t / 10)

        return u_base + u_var, v_base + v_var

    def initial_condition(self, x, y):
        """Enhanced initial conditions"""
        pm25 = self.pollutants['PM2.5']['background'] * torch.ones_like(x)
        pm10 = self.pollutants['PM10']['background'] * torch.ones_like(x)

        # Enhanced pollution sources
        sources = [
            (self.delhi_x, self.delhi_y, 45.0, 70.0),   # Delhi center
            (10, 10, 35.0, 55.0),   # Industrial area 1
            (40, 30, 30.0, 50.0),   # Industrial area 2
            (5, 35, 20.0, 35.0),    # Residential area
            (45, 5, 25.0, 40.0),    # Transportation hub
        ]

        for sx, sy, intensity_pm25, intensity_pm10 in sources:
            distance = torch.sqrt((x - sx)**2 + (y - sy)**2)
            pm25 += intensity_pm25 * torch.exp(-distance**2 / 60.0)
            pm10 += intensity_pm10 * torch.exp(-distance**2 / 60.0)

        return torch.cat([pm25, pm10], dim=1)

    def physics_loss(self, model, x, y, t):
        """Physics loss for both pollutants"""
        x.requires_grad = True
        y.requires_grad = True
        t.requires_grad = True

        # Get predictions
        C = model(x, y, t)
        C_pm25 = C[:, 0:1]
        C_pm10 = C[:, 1:2]

        # Compute gradients for PM2.5
        C_pm25_t = torch.autograd.grad(C_pm25, t, grad_outputs=torch.ones_like(C_pm25), create_graph=True)[0]
        C_pm25_x = torch.autograd.grad(C_pm25, x, grad_outputs=torch.ones_like(C_pm25), create_graph=True)[0]
        C_pm25_y = torch.autograd.grad(C_pm25, y, grad_outputs=torch.ones_like(C_pm25), create_graph=True)[0]
        C_pm25_xx = torch.autograd.grad(C_pm25_x, x, grad_outputs=torch.ones_like(C_pm25_x), create_graph=True)[0]
        C_pm25_yy = torch.autograd.grad(C_pm25_y, y, grad_outputs=torch.ones_like(C_pm25_y), create_graph=True)[0]

        # Compute gradients for PM10
        C_pm10_t = torch.autograd.grad(C_pm10, t, grad_outputs=torch.ones_like(C_pm10), create_graph=True)[0]
        C_pm10_x = torch.autograd.grad(C_pm10, x, grad_outputs=torch.ones_like(C_pm10), create_graph=True)[0]
        C_pm10_y = torch.autograd.grad(C_pm10, y, grad_outputs=torch.ones_like(C_pm10), create_graph=True)[0]
        C_pm10_xx = torch.autograd.grad(C_pm10_x, x, grad_outputs=torch.ones_like(C_pm10_x), create_graph=True)[0]
        C_pm10_yy = torch.autograd.grad(C_pm10_y, y, grad_outputs=torch.ones_like(C_pm10_y), create_graph=True)[0]

        # Get wind field
        u, v = self.wind_field(x, y, t)

        # Source terms
        S_pm25 = self.source_term(x, y, t, 'PM2.5')
        S_pm10 = self.source_term(x, y, t, 'PM10')

        # Decay terms
        decay_pm25 = self.pollutants['PM2.5']['decay_rate'] * C_pm25
        decay_pm10 = self.pollutants['PM10']['decay_rate'] * C_pm10

        # PDE residuals
        residual_pm25 = (C_pm25_t + u * C_pm25_x + v * C_pm25_y -
                        self.pollutants['PM2.5']['D'] * (C_pm25_xx + C_pm25_yy) -
                        S_pm25 + decay_pm25)

        residual_pm10 = (C_pm10_t + u * C_pm10_x + v * C_pm10_y -
                        self.pollutants['PM10']['D'] * (C_pm10_xx + C_pm10_yy) -
                        S_pm10 + decay_pm10)

        return torch.mean(residual_pm25**2), torch.mean(residual_pm10**2)

    def train_model(self, epochs=4000, lr=0.001):
        """Enhanced training with 4000 epochs"""
        print("="*80)
        print("ENHANCED DUAL POLLUTANT PINN TRAINING")
        print(f"Epochs: {epochs} | Time Domain: {self.t_max} days")
        print("="*80)

        model = EnhancedDualPollutantPINN().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=300, factor=0.5)

        # Generate training data
        n_samples = 15000  # Increased for better coverage
        x = torch.rand(n_samples, 1).to(self.device) * (self.x_max - self.x_min) + self.x_min
        y = torch.rand(n_samples, 1).to(self.device) * (self.y_max - self.y_min) + self.y_min
        t = torch.rand(n_samples, 1).to(self.device) * (self.t_max - self.t_min) + self.t_min

        # Boundary conditions
        n_bc = 3000
        x_bc = torch.rand(n_bc, 1).to(self.device) * (self.x_max - self.x_min) + self.x_min
        y_bc = torch.rand(n_bc, 1).to(self.device) * (self.y_max - self.y_min) + self.y_min
        t_bc = torch.zeros_like(x_bc)
        C0_bc = self.initial_condition(x_bc, y_bc)

        # Clear loss histories
        self.total_losses = []
        self.physics_losses = []
        self.boundary_losses = []

        # Training loop
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Physics loss
            loss_physics_pm25, loss_physics_pm10 = self.physics_loss(model, x, y, t)
            loss_physics = loss_physics_pm25 + loss_physics_pm10

            # Boundary loss
            C_pred = model(x_bc, y_bc, t_bc)
            loss_boundary = torch.mean((C_pred - C0_bc)**2)

            # Total loss
            loss = loss_physics + 10.0 * loss_boundary
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            # Store losses
            self.total_losses.append(loss.item())
            self.physics_losses.append(loss_physics.item())
            self.boundary_losses.append(loss_boundary.item())

            if epoch % 500 == 0:
                print(f"Epoch {epoch:4d}/{epochs}: Total={loss.item():.6f}, "
                      f"Physics={loss_physics.item():.6f}, "
                      f"Boundary={loss_boundary.item():.6f}")

        print(f"\nTraining completed: Final Loss = {self.total_losses[-1]:.6f}")
        return model

    def generate_visualization_grid(self, resolution=100):
        """Create grid for visualization"""
        x_grid = np.linspace(self.x_min, self.x_max, resolution)
        y_grid = np.linspace(self.y_min, self.y_max, resolution)
        return np.meshgrid(x_grid, y_grid)

    def predict_concentrations(self, model, X, Y, t_day):
        """Predict concentrations for both pollutants"""
        x_tensor = torch.FloatTensor(X.flatten()[:, None]).to(self.device)
        y_tensor = torch.FloatTensor(Y.flatten()[:, None]).to(self.device)
        t_tensor = torch.ones_like(x_tensor).to(self.device) * t_day

        with torch.no_grad():
            C = model(x_tensor, y_tensor, t_tensor)
            C_pm25 = C[:, 0].cpu().numpy().reshape(X.shape)
            C_pm10 = C[:, 1].cpu().numpy().reshape(X.shape)

        return C_pm25, C_pm10

# %% [markdown]
# ## VISUALIZATION FUNCTIONS

# %% [code]
class VisualizationSuite:
    """Complete visualization suite for 60-day analysis"""

    def __init__(self, model, simulation):
        self.model = model
        self.sim = simulation
        self.X, self.Y = simulation.generate_visualization_grid(120)  # Higher resolution

    # ====================== GRAPH 1: LOSS CONVERGENCE ======================
    def plot_loss_convergence(self):
        """Plot total, physics, and boundary losses"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Total Loss
        axes[0].plot(self.sim.total_losses, 'b-', linewidth=2)
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Training Loss Convergence', fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Physics Loss
        axes[1].plot(self.sim.physics_losses, 'r-', linewidth=2)
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Physics Loss')
        axes[1].set_title('Physics Loss Convergence', fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Boundary Loss
        axes[2].plot(self.sim.boundary_losses, 'g-', linewidth=2)
        axes[2].set_yscale('log')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Boundary Loss')
        axes[2].set_title('Boundary Condition Loss Convergence', fontweight='bold')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('loss_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ Saved: loss_convergence.png")

    # ====================== GRAPH 2: POLLUTION SPREAD OVER TIME ======================
    def plot_pollution_spread_over_time(self):
        """60-day pollution spread with 5-day intervals"""
        print("\nGenerating 60-day pollution spread analysis...")

        # Create time points (every 5 days from 0 to 60)
        time_points = list(range(0, 61, 5))
        n_rows = 4
        n_cols = 4

        # Create figure for PM2.5
        fig_pm25, axes_pm25 = plt.subplots(n_rows, n_cols, figsize=(20, 16))
        fig_pm25.suptitle('PM2.5 Pollution Spread Over 60 Days: Aravalli Demolition Impact\n(5-Day Intervals)',
                         fontsize=16, fontweight='bold', y=1.02)

        # Create figure for PM10
        fig_pm10, axes_pm10 = plt.subplots(n_rows, n_cols, figsize=(20, 16))
        fig_pm10.suptitle('PM10 Pollution Spread Over 60 Days: Aravalli Demolition Impact\n(5-Day Intervals)',
                         fontsize=16, fontweight='bold', y=1.02)

        # Create pollution colormap
        pollution_cmap = LinearSegmentedColormap.from_list(
            'pollution', ['#00ff00', '#ffff00', '#ff8000', '#ff0000', '#800080']
        )

        vmax_pm25 = 0
        vmax_pm10 = 0

        # First pass to find max values for consistent color scaling
        for t_day in time_points:
            C_pm25, C_pm10 = self.sim.predict_concentrations(self.model, self.X, self.Y, t_day)
            vmax_pm25 = max(vmax_pm25, np.max(C_pm25))
            vmax_pm10 = max(vmax_pm10, np.max(C_pm10))

        # Plot each time point
        for idx, t_day in enumerate(time_points):
            row = idx // n_cols
            col = idx % n_cols

            C_pm25, C_pm10 = self.sim.predict_concentrations(self.model, self.X, self.Y, t_day)

            # Plot PM2.5
            ax_pm25 = axes_pm25[row, col]
            im_pm25 = ax_pm25.contourf(self.X, self.Y, C_pm25, levels=50,
                                      cmap=pollution_cmap, vmin=20, vmax=vmax_pm25)
            self._add_landmarks(ax_pm25)
            ax_pm25.set_title(f'Day {t_day}', fontsize=10, fontweight='bold')
            ax_pm25.set_xlabel('East-West (km)' if row == n_rows-1 else '')
            ax_pm25.set_ylabel('North-South (km)' if col == 0 else '')

            # Plot PM10
            ax_pm10 = axes_pm10[row, col]
            im_pm10 = ax_pm10.contourf(self.X, self.Y, C_pm10, levels=50,
                                      cmap=pollution_cmap, vmin=30, vmax=vmax_pm10)
            self._add_landmarks(ax_pm10)
            ax_pm10.set_title(f'Day {t_day}', fontsize=10, fontweight='bold')
            ax_pm10.set_xlabel('East-West (km)' if row == n_rows-1 else '')
            ax_pm10.set_ylabel('North-South (km)' if col == 0 else '')

        # Add colorbars
        fig_pm25.colorbar(im_pm25, ax=axes_pm25.ravel().tolist(),
                         orientation='horizontal', pad=0.02, aspect=50, label='PM2.5 (μg/m³)')
        fig_pm10.colorbar(im_pm10, ax=axes_pm10.ravel().tolist(),
                         orientation='horizontal', pad=0.02, aspect=50, label='PM10 (μg/m³)')

        plt.tight_layout()
        fig_pm25.savefig('pm25_spread_60days.png', dpi=300, bbox_inches='tight')
        fig_pm10.savefig('pm10_spread_60days.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Saved: pm25_spread_60days.png")
        print("✅ Saved: pm10_spread_60days.png")

    # ====================== GRAPH 3: COMPREHENSIVE ANALYSIS ======================
    def plot_comprehensive_analysis(self):
        """Comprehensive dashboard combining PM2.5 and PM10"""
        print("\nGenerating comprehensive analysis dashboard...")

        # Get data for key days
        C_pm25_day0, C_pm10_day0 = self.sim.predict_concentrations(self.model, self.X, self.Y, 0)
        C_pm25_day15, C_pm10_day15 = self.sim.predict_concentrations(self.model, self.X, self.Y, 15)
        C_pm25_day30, C_pm10_day30 = self.sim.predict_concentrations(self.model, self.X, self.Y, 30)
        C_pm25_day60, C_pm10_day60 = self.sim.predict_concentrations(self.model, self.X, self.Y, 60)

        # Calculate differences
        diff_pm25_day15 = C_pm25_day15 - C_pm25_day0
        diff_pm10_day15 = C_pm10_day15 - C_pm10_day0
        diff_pm25_day30 = C_pm25_day30 - C_pm25_day0
        diff_pm10_day30 = C_pm10_day30 - C_pm10_day0

        # Create figure
        fig = plt.figure(figsize=(22, 16))
        gs = fig.add_gridspec(4, 5, height_ratios=[1, 1, 0.8, 0.8])

        # Colormaps
        conc_cmap = LinearSegmentedColormap.from_list('conc', ['#00ff00', '#ffff00', '#ff8000', '#ff0000'])
        diff_cmap = LinearSegmentedColormap.from_list('diff', ['blue', 'white', 'red'])

        # Row 1: PM2.5 Analysis
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.contourf(self.X, self.Y, C_pm25_day0, levels=50, cmap=conc_cmap)
        ax1.set_title('PM2.5 Baseline (Day 0)', fontweight='bold')
        ax1.set_ylabel('North-South (km)')
        plt.colorbar(im1, ax=ax1, label='μg/m³')

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.contourf(self.X, self.Y, C_pm25_day15, levels=50, cmap=conc_cmap)
        ax2.set_title('PM2.5 Peak Impact (Day 15)', fontweight='bold')
        plt.colorbar(im2, ax=ax2, label='μg/m³')

        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.contourf(self.X, self.Y, diff_pm25_day15, levels=50, cmap=diff_cmap)
        ax3.set_title('PM2.5 Increase (Day 15)', fontweight='bold')
        plt.colorbar(im3, ax=ax3, label='Δμg/m³')

        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.contourf(self.X, self.Y, C_pm25_day30, levels=50, cmap=conc_cmap)
        ax4.set_title('PM2.5 Mid-Term (Day 30)', fontweight='bold')
        plt.colorbar(im4, ax=ax4, label='μg/m³')

        ax5 = fig.add_subplot(gs[0, 4])
        im5 = ax5.contourf(self.X, self.Y, C_pm25_day60, levels=50, cmap=conc_cmap)
        ax5.set_title('PM2.5 Recovery (Day 60)', fontweight='bold')
        plt.colorbar(im5, ax=ax5, label='μg/m³')

        # Row 2: PM10 Analysis
        ax6 = fig.add_subplot(gs[1, 0])
        im6 = ax6.contourf(self.X, self.Y, C_pm10_day0, levels=50, cmap=conc_cmap)
        ax6.set_title('PM10 Baseline (Day 0)', fontweight='bold')
        ax6.set_xlabel('East-West (km)')
        ax6.set_ylabel('North-South (km)')
        plt.colorbar(im6, ax=ax6, label='μg/m³')

        ax7 = fig.add_subplot(gs[1, 1])
        im7 = ax7.contourf(self.X, self.Y, C_pm10_day15, levels=50, cmap=conc_cmap)
        ax7.set_title('PM10 Peak Impact (Day 15)', fontweight='bold')
        ax7.set_xlabel('East-West (km)')
        plt.colorbar(im7, ax=ax7, label='μg/m³')

        ax8 = fig.add_subplot(gs[1, 2])
        im8 = ax8.contourf(self.X, self.Y, diff_pm10_day15, levels=50, cmap=diff_cmap)
        ax8.set_title('PM10 Increase (Day 15)', fontweight='bold')
        ax8.set_xlabel('East-West (km)')
        plt.colorbar(im8, ax=ax8, label='Δμg/m³')

        ax9 = fig.add_subplot(gs[1, 3])
        im9 = ax9.contourf(self.X, self.Y, C_pm10_day30, levels=50, cmap=conc_cmap)
        ax9.set_title('PM10 Mid-Term (Day 30)', fontweight='bold')
        ax9.set_xlabel('East-West (km)')
        plt.colorbar(im9, ax=ax9, label='μg/m³')

        ax10 = fig.add_subplot(gs[1, 4])
        im10 = ax10.contourf(self.X, self.Y, C_pm10_day60, levels=50, cmap=conc_cmap)
        ax10.set_title('PM10 Recovery (Day 60)', fontweight='bold')
        ax10.set_xlabel('East-West (km)')
        plt.colorbar(im10, ax=ax10, label='μg/m³')

        # Add landmarks to all subplots
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]:
            self._add_landmarks(ax)

        # Row 3: Statistics Panel
        ax_stats = fig.add_subplot(gs[2, :])
        ax_stats.axis('off')

        # Calculate statistics
        stats_text = self._generate_statistics_text(diff_pm25_day15, diff_pm10_day15,
                                                   diff_pm25_day30, diff_pm10_day30)

        ax_stats.text(0.02, 0.98, stats_text, fontsize=10, family='monospace',
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # Row 4: Combined Time Series
        ax_ts = fig.add_subplot(gs[3, :])
        time_points = np.linspace(0, 60, 121)
        pm25_ts, pm10_ts = [], []

        for t in time_points:
            C_pm25, C_pm10 = self.sim.predict_concentrations(self.model,
                                                           np.array([[self.sim.delhi_x]]),
                                                           np.array([[self.sim.delhi_y]]),
                                                           t)
            pm25_ts.append(C_pm25[0][0])
            pm10_ts.append(C_pm10[0][0])

        ax_ts.plot(time_points, pm25_ts, 'b-', linewidth=2.5, label='PM2.5')
        ax_ts.plot(time_points, pm10_ts, 'r-', linewidth=2.5, label='PM10')
        ax_ts.fill_between(time_points, pm25_ts, self.sim.pollutants['PM2.5']['background'],
                          alpha=0.2, color='blue')
        ax_ts.fill_between(time_points, pm10_ts, self.sim.pollutants['PM10']['background'],
                          alpha=0.2, color='red')

        # Add safety thresholds
        ax_ts.axhline(y=self.sim.pollutants['PM2.5']['danger_level'], color='blue',
                     linestyle='--', alpha=0.5, label='PM2.5 Danger Level')
        ax_ts.axhline(y=self.sim.pollutants['PM10']['danger_level'], color='red',
                     linestyle='--', alpha=0.5, label='PM10 Danger Level')
        ax_ts.axhline(y=self.sim.pollutants['PM2.5']['safe_level'], color='blue',
                     linestyle=':', alpha=0.3, label='PM2.5 Safe Level')
        ax_ts.axhline(y=self.sim.pollutants['PM10']['safe_level'], color='red',
                     linestyle=':', alpha=0.3, label='PM10 Safe Level')

        ax_ts.set_xlabel('Time (days)', fontsize=11)
        ax_ts.set_ylabel('Concentration (μg/m³)', fontsize=11)
        ax_ts.set_title('Pollution Timeline at Delhi Center: 60-Day Evolution',
                       fontsize=12, fontweight='bold')
        ax_ts.legend(loc='upper right', ncol=2)
        ax_ts.grid(True, alpha=0.3)

        plt.suptitle('Comprehensive Analysis: Aravalli Demolition Impact on Delhi Air Quality\n'
                    'Dual Pollutant PINN Simulation | 60-Day Analysis | 4000 Epochs Training',
                    fontsize=18, fontweight='bold', y=1.0)

        plt.tight_layout()
        plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Saved: comprehensive_analysis.png")

    # ====================== GRAPH 4: DISTRIBUTION OF POLLUTION INCREASE ======================
    def plot_pollution_distribution(self):
        """Distribution of pollution increases"""
        print("\nGenerating pollution distribution analysis...")

        # Get data for Day 15 (peak impact)
        C_pm25_day0, C_pm10_day0 = self.sim.predict_concentrations(self.model, self.X, self.Y, 0)
        C_pm25_day15, C_pm10_day15 = self.sim.predict_concentrations(self.model, self.X, self.Y, 15)
        diff_pm25 = C_pm25_day15 - C_pm25_day0
        diff_pm10 = C_pm10_day15 - C_pm10_day0

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # PM2.5 distribution
        axes[0].hist(diff_pm25.flatten(), bins=80, alpha=0.7, color='blue',
                    edgecolor='black', density=True)
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=2)
        axes[0].axvline(x=20, color='red', linestyle='--', linewidth=2, label='Significant (20)')
        axes[0].axvline(x=40, color='darkred', linestyle='--', linewidth=2, label='Severe (40)')
        axes[0].set_xlabel('PM2.5 Increase (μg/m³)', fontsize=11)
        axes[0].set_ylabel('Probability Density', fontsize=11)
        axes[0].set_title('Distribution of PM2.5 Increases (Day 15)', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # PM10 distribution
        axes[1].hist(diff_pm10.flatten(), bins=80, alpha=0.7, color='orange',
                    edgecolor='black', density=True)
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=2)
        axes[1].axvline(x=30, color='red', linestyle='--', linewidth=2, label='Significant (30)')
        axes[1].axvline(x=60, color='darkred', linestyle='--', linewidth=2, label='Severe (60)')
        axes[1].set_xlabel('PM10 Increase (μg/m³)', fontsize=11)
        axes[1].set_ylabel('Probability Density', fontsize=11)
        axes[1].set_title('Distribution of PM10 Increases (Day 15)', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('pollution_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Saved: pollution_distribution.png")

    # ====================== GRAPH 5: SPREAD RATE COMPARISON ======================
    def plot_spread_rate_comparison(self):
        """Side-by-side comparison of PM2.5 and PM10 spread rates"""
        print("\nGenerating spread rate comparison...")

        # Calculate pollution fronts over time
        days_to_analyze = list(range(0, 61, 3))
        pm25_fronts = []
        pm10_fronts = []

        for t_day in days_to_analyze:
            C_pm25, C_pm10 = self.sim.predict_concentrations(self.model, self.X, self.Y, t_day)
            # Calculate distance where concentration > baseline + 10
            pm25_front = self._calculate_pollution_front(C_pm25, self.sim.pollutants['PM2.5']['background'] + 10)
            pm10_front = self._calculate_pollution_front(C_pm10, self.sim.pollutants['PM10']['background'] + 15)
            pm25_fronts.append(pm25_front)
            pm10_fronts.append(pm10_front)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Spread distance over time
        axes[0].plot(days_to_analyze, pm25_fronts, 'b-o', linewidth=2, markersize=4, label='PM2.5')
        axes[0].plot(days_to_analyze, pm10_fronts, 'r-s', linewidth=2, markersize=4, label='PM10')
        axes[0].set_xlabel('Time (days)', fontsize=11)
        axes[0].set_ylabel('Pollution Front Distance from Aravalli (km)', fontsize=11)
        axes[0].set_title('Pollution Spread Distance Over Time', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Spread rate (derivative)
        pm25_rate = np.gradient(pm25_fronts, days_to_analyze)
        pm10_rate = np.gradient(pm10_fronts, days_to_analyze)

        axes[1].plot(days_to_analyze[1:-1], pm25_rate[1:-1], 'b-o', linewidth=2, markersize=4, label='PM2.5 Rate')
        axes[1].plot(days_to_analyze[1:-1], pm10_rate[1:-1], 'r-s', linewidth=2, markersize=4, label='PM10 Rate')
        axes[1].set_xlabel('Time (days)', fontsize=11)
        axes[1].set_ylabel('Spread Rate (km/day)', fontsize=11)
        axes[1].set_title('Pollution Spread Rate Over Time', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('PM2.5 vs PM10 Spread Rate Comparison: Aravalli Demolition Impact',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('spread_rate_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ Saved: spread_rate_comparison.png")

        # Print spread statistics
        print(f"\n📊 Spread Statistics:")
        print(f"PM2.5 Max spread distance: {max(pm25_fronts):.1f} km")
        print(f"PM10 Max spread distance: {max(pm10_fronts):.1f} km")
        print(f"PM2.5 Avg spread rate: {np.mean(pm25_rate[1:-1]):.2f} km/day")
        print(f"PM10 Avg spread rate: {np.mean(pm10_rate[1:-1]):.2f} km/day")
        print(f"PM10 spreads {max(pm10_fronts)/max(pm25_fronts):.2f}x faster than PM2.5")

    # ====================== HELPER FUNCTIONS ======================
    def _add_landmarks(self, ax):
        """Add landmarks to plot"""
        ax.scatter(self.sim.aravalli_x, self.sim.aravalli_y, c='black', marker='^',
                  s=180, label='Aravalli Hills', edgecolors='white', linewidth=1.5)
        ax.scatter(self.sim.delhi_x, self.sim.delhi_y, c='blue', marker='o',
                  s=120, label='Delhi Center', edgecolors='white', linewidth=1.5)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2)

    def _generate_statistics_text(self, diff_pm25_day15, diff_pm10_day15,
                                 diff_pm25_day30, diff_pm10_day30):
        """Generate statistics text for dashboard"""
        # Calculate areas affected
        thresholds = [10, 20, 30, 40, 50]
        stats_lines = []

        stats_lines.append(" " * 20 + "STATISTICAL SUMMARY")
        stats_lines.append("=" * 60)
        stats_lines.append("PM2.5 ANALYSIS (Day 15 Peak):")
        stats_lines.append(f"  Max increase: {np.max(diff_pm25_day15):.1f} μg/m³")
        stats_lines.append(f"  Avg increase: {np.mean(diff_pm25_day15):.1f} μg/m³")
        stats_lines.append(f"  Delhi Center: {diff_pm25_day15[60, 60]:.1f} μg/m³")

        for thresh in thresholds:
            area = np.sum(diff_pm25_day15 > thresh) / diff_pm25_day15.size * 100
            stats_lines.append(f"  Area >{thresh:2d} μg/m³: {area:5.1f}%")

        stats_lines.append("\nPM10 ANALYSIS (Day 15 Peak):")
        stats_lines.append(f"  Max increase: {np.max(diff_pm10_day15):.1f} μg/m³")
        stats_lines.append(f"  Avg increase: {np.mean(diff_pm10_day15):.1f} μg/m³")
        stats_lines.append(f"  Delhi Center: {diff_pm10_day15[60, 60]:.1f} μg/m³")

        for thresh in thresholds:
            area = np.sum(diff_pm10_day15 > thresh) / diff_pm10_day15.size * 100
            stats_lines.append(f"  Area >{thresh:2d} μg/m³: {area:5.1f}%")

        stats_lines.append("\nRECOVERY ANALYSIS (Day 30):")
        stats_lines.append(f"  PM2.5 remaining: {np.mean(diff_pm25_day30)/np.mean(diff_pm25_day15)*100:.1f}%")
        stats_lines.append(f"  PM10 remaining: {np.mean(diff_pm10_day30)/np.mean(diff_pm10_day15)*100:.1f}%")

        return "\n".join(stats_lines)

    def _calculate_pollution_front(self, concentration, threshold):
        """Calculate distance of pollution front from Aravalli"""
        # Find all points above threshold
        above_threshold = concentration > threshold

        if not np.any(above_threshold):
            return 0

        # Calculate distances from Aravalli for points above threshold
        distances = []
        for i in range(concentration.shape[0]):
            for j in range(concentration.shape[1]):
                if above_threshold[i, j]:
                    dist = np.sqrt((self.X[i, j] - self.sim.aravalli_x)**2 +
                                  (self.Y[i, j] - self.sim.aravalli_y)**2)
                    distances.append(dist)

        return np.max(distances) if distances else 0

# %% [markdown]
# ## MAIN EXECUTION

# %% [code]
def main():
    """Main execution function"""
    print("="*80)
    print("ENHANCED ARAVALLI DEMOLITION IMPACT SIMULATION")
    print("60-Day Analysis | 4000 Epochs | Dual Pollutant (PM2.5 & PM10)")
    print("="*80)

    try:
        # Initialize simulation
        simulation = EnhancedAravalliModel(device=device)

        # Train model with 4000 epochs
        print("\n🚀 Starting training with 4000 epochs...")
        model = simulation.train_model(epochs=4000, lr=0.001)

        # Initialize visualization suite
        viz = VisualizationSuite(model, simulation)

        # Generate all requested graphs
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        # 1. Loss Convergence Plots
        viz.plot_loss_convergence()

        # 2. Pollution Spread Over Time (60 days, 5-day intervals)
        viz.plot_pollution_spread_over_time()

        # 3. Comprehensive Analysis Dashboard
        viz.plot_comprehensive_analysis()

        # 4. Distribution of Pollution Increase
        viz.plot_pollution_distribution()

        # 5. Spread Rate Comparison
        viz.plot_spread_rate_comparison()

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'simulation_params': {
                'days': 60,
                'epochs': 4000,
                'pollutants': simulation.pollutants
            }
        }, 'enhanced_dual_pollutant_model.pth')

        print("\n" + "="*80)
        print("✅ SIMULATION COMPLETE!")
        print("="*80)
        print("\n📁 Generated Files:")
        print("1. loss_convergence.png - Training loss plots")
        print("2. pm25_spread_60days.png - PM2.5 spread over 60 days")
        print("3. pm10_spread_60days.png - PM10 spread over 60 days")
        print("4. comprehensive_analysis.png - Complete dashboard")
        print("5. pollution_distribution.png - Distribution histograms")
        print("6. spread_rate_comparison.png - Spread rate comparison")
        print("7. enhanced_dual_pollutant_model.pth - Saved model")
        print("\n📊 For Thesis Abstract:")
        print("Use statistics from comprehensive_analysis.png")
        print("="*80)

        return model, simulation

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# %% [code]
if __name__ == "__main__":
    # Run the enhanced simulation
    model, simulation = main()