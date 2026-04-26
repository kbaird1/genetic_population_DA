import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys

# Ensure the src folder is in the path
sys.path.append(os.path.abspath('./'))
from src.kalman_filter import StandardKalmanFilter

def generate_dashboard(data_path, topology_name="Simulation"):
    """
    Generates an interactive Plotly dashboard demonstrating the impact 
    of spatial structure on genetic drift.
    """
    # 1. Load Data
    data = np.load(data_path)
    true_I = data['true_I']
    T, K, original_L = true_I.shape
    grid_size = int(np.sqrt(K))

    # 2. Run Kalman Filter
    kf = StandardKalmanFilter(data_path)
    best_Ne, _ = kf.estimate_Ne()
    kf_trajectory = kf.get_filtered_trajectory(best_Ne)
    true_peak = kf.peak_true_infections
    
    # Calculate the Drift Multiplier
    drift_multiplier = true_peak / best_Ne

    # 3. Process Ground Truth Data (with Coarse Graining)
    true_national_counts = np.sum(true_I, axis=1)
    coarse_true_counts = kf._coarse_grain(true_national_counts, min_freq_threshold=0.01)
    total_I_per_day = np.sum(coarse_true_counts, axis=1, keepdims=True)
    total_I_per_day[total_I_per_day == 0] = 1e-9 
    true_freqs = coarse_true_counts / total_I_per_day
    
    L = true_freqs.shape[1] 

    # Calculate additional metrics for new plots
    tvd_error = 0.5 * np.sum(np.abs(true_freqs - kf_trajectory), axis=1)
    true_daily_infections = np.sum(true_national_counts, axis=1)
    
    # Active demes per day: Count demes where total infections > 0
    active_demes_per_day = np.sum(np.sum(true_I, axis=2) > 0, axis=1)

    # Pre-compute Grid Data
    grid_data_frames = [np.sum(true_I[t], axis=1).reshape(grid_size, grid_size) for t in range(T)]

    # 4. Build Plotly Figure (2 Rows, 3 Columns)
    fig = make_subplots(
        rows=2, cols=3, 
        subplot_titles=(
            f"True Spatial Spread ({topology_name})", 
            "National Allele Frequencies", 
            "Model Error (TVD)",
            "True Pop vs Inferred Ne",
            "Active Demes",
            "" # Empty slot or you can put another metric here
        ),
        specs=[[{"type": "heatmap"}, {"type": "bar"}, {"type": "scatter"}],
               [{"type": "scatter", "colspan": 2}, None, {"type": "scatter"}]],
        row_heights=[0.5, 0.5]
    )

    # Set up colors: Distinct grey for the "Other" (coarse-grained) bin
    colors = ['skyblue'] * (L - 1) + ['lightgrey']

    # --- ROW 1 INITIAL TRACES ---
    fig.add_trace(go.Heatmap(z=grid_data_frames[0], colorscale='inferno', zmin=0, zmax=50, showscale=False), row=1, col=1)
    fig.add_trace(go.Bar(x=np.arange(L), y=true_freqs[0], name="True Freq", marker_color=colors), row=1, col=2)
    fig.add_trace(go.Scatter(x=np.arange(L), y=kf_trajectory[0], mode='markers', marker_symbol='x', marker_color='red', marker_size=8, name=f"KF Estimate"), row=1, col=2)
    fig.add_trace(go.Scatter(x=np.arange(T), y=tvd_error, mode='lines', line=dict(color='black', width=2), showlegend=False), row=1, col=3)
    fig.add_trace(go.Scatter(x=[0], y=[tvd_error[0]], mode='markers', marker=dict(color='red', size=10), showlegend=False), row=1, col=3)

    # --- ROW 2 INITIAL TRACES ---
    # The 'Gotcha' Plot
    fig.add_trace(go.Scatter(x=np.arange(T), y=true_daily_infections, mode='lines', name='True Infections (N)', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=[0, T-1], y=[best_Ne, best_Ne], mode='lines', name='Inferred Ne', line=dict(color='red', dash='dash')), row=2, col=1)
    
    # Active Demes Plot
    fig.add_trace(go.Scatter(x=np.arange(T), y=active_demes_per_day, mode='lines', name='Active Demes', line=dict(color='purple')), row=2, col=3)

    # 5. Build Frames for Slider
    frames = []
    for t in range(T):
        frames.append(go.Frame(
            data=[
                go.Heatmap(z=grid_data_frames[t]),
                go.Bar(y=true_freqs[t]),
                go.Scatter(y=kf_trajectory[t]),
                go.Scatter(x=np.arange(T), y=tvd_error), 
                go.Scatter(x=[t], y=[tvd_error[t]]),
                # Row 2 traces don't need to animate point-by-point, they are static lines
                go.Scatter(x=np.arange(T), y=true_daily_infections),
                go.Scatter(x=[0, T-1], y=[best_Ne, best_Ne]),
                go.Scatter(x=np.arange(T), y=active_demes_per_day)
            ],
            name=f"Day {t}"
        ))

    fig.frames = frames

    # 6. Add Slider and Layout Polish
    steps = []
    for t in range(T):
        step = dict(
            method="animate",
            args=[[f"Day {t}"], dict(mode="immediate", frame=dict(duration=100, redraw=True), transition=dict(duration=0))],
            label=str(t)
        )
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Simulation Day: "},
        pad={"t": 50},
        steps=steps
    )]

    # Emphasize the Drift Multiplier in the Title
    fig.update_layout(
        sliders=sliders,
        height=800,
        width=1400,
        title_text=f"<b>{topology_name.upper()} Topology</b> | Spatial structure inflated genetic drift by <b>{drift_multiplier:.1f}x</b> (Peak N: {true_peak} vs Ne: {best_Ne:.0f})",
        title_font_size=20,
        barmode='overlay',
        plot_bgcolor='white',
        showlegend=True
    )

    # Axes styling
    fig.update_yaxes(range=[0, 1.0], row=1, col=2)
    
    # Set the 'Gotcha' plot to log scale to show the massive gap clearly
    fig.update_yaxes(type="log", title_text="Individuals (Log Scale)", row=2, col=1)
    fig.update_yaxes(title_text="Active Demes", row=2, col=3)

    return fig

if __name__ == "__main__":
    # Ensure this path points to your actual generated data
    data_path = "./data/raw/sim_output.npz" 
    
    if os.path.exists(data_path):
        fig = generate_dashboard(data_path, topology_name="Global Jackpot Structured")
        fig.show()
    else:
        print(f"Error: Could not find data file at {data_path}")