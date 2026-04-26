import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import mplcursors  # New library for interactivity

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_sweep():
    try:
        with open("./data/experiments/resolution_sweep.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Data file not found. Run the sweep script first.")
        return

    metrics_to_plot = ["RMSE", "TVD", "JS_Distance", "KL_Divergence"]
    colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c']

    for sc_name, results in data.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Resolution Sweep: {sc_name.replace('_', ' ')}\n(Left side = Perfect/Peak Sequencing, Right side = Noisy)", 
                     fontsize=16, fontweight='bold')
        
        ct_vals = [r["c_t"] for r in results]
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes.flatten()[i]
            y_vals = [r["metrics"][metric] for r in results]
            
            # Plot the line
            ax.plot(ct_vals, y_vals, '--', color=colors[i], alpha=0.4)
            
            # Plot the dots and capture the object (sc) for the cursor
            sc = ax.scatter(ct_vals, y_vals, color=colors[i], s=60, edgecolor='black', zorder=3)
            
            ax.set_title(f"Avg {metric}")
            ax.set_xlabel("Sequencing Capacity ($c_t$) - Log Scale")
            ax.set_ylabel("Error Value")
            ax.set_xscale('log')
            ax.set_xlim(max(ct_vals), 1) 
        
        # for i, metric in enumerate(metrics_to_plot):
        #     ax = axes.flatten()[i]
        #     y_vals = [r["metrics"][metric] for r in results]
            
        #     # 1. Keep the log scale for spacing (so small values aren't bunched up)
        #     ax.set_xscale('log')
            
        #     # 2. Plot the data
        #     ax.plot(ct_vals, y_vals, '--', color=colors[i], alpha=0.4)
        #     sc = ax.scatter(ct_vals, y_vals, color=colors[i], s=60, edgecolor='black', zorder=3)
            
        #     # 3. CRITICAL CHANGE: Force ticks to be exactly your ct_vals
        #     ax.set_xticks(ct_vals)
            
        #     # 4. Use a ScalarFormatter to keep numbers as "100" instead of "10^2"
        #     from matplotlib.ticker import ScalarFormatter
        #     ax.xaxis.set_major_formatter(ScalarFormatter())
            
        #     # Optional: Rotate labels if you have many points and they overlap
        #     # plt.setp(ax.get_xticklabels(), rotation=45)

        #     ax.set_title(f"Avg {metric}")
        #     ax.set_xlabel("Sequencing Capacity ($c_t$)")
        #     ax.set_ylabel("Error Value")
            
        #     # Keep your reversed logic if you want high-capacity on the left
        #     ax.set_xlim(max(ct_vals), min(ct_vals))

            # Create the hover interaction for this specific subplot
            cursor = mplcursors.cursor(sc, hover=True)
            
            # This function formats what you see when you hover
            @cursor.connect("add")
            def on_add(sel, m_name=metric):
                # sel.target[0] is the x value (ct), sel.target[1] is the y value (error)
                sel.annotation.set_text(f"Metric: {m_name}\nc_t: {int(sel.target[0])}\nValue: {sel.target[1]:.4f}")
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

            # Format X-axis ticks to be readable numbers instead of scientific notation
            from matplotlib.ticker import ScalarFormatter
            ax.xaxis.set_major_formatter(ScalarFormatter())

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"./data/experiments/sweep_{sc_name}.png"
        plt.savefig(filename, dpi=300)
        print(f"Saved: {filename}. Interactive window opening...")
        
        # NOTE: Interaction only works while the window is open on your desktop.
        # It won't work in the saved .png file.
        plt.show()

if __name__ == "__main__":
    plot_sweep()