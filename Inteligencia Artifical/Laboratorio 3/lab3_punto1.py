"""
Laboratorio N°3 - Punto 1
Aprendizaje por Refuerzo: Frozen Lake
Parte A: Value Iteration (basado en modelo)
Parte B: Q-Learning (libre de modelo)
"""

from IPython.core import crashhandler
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore")

# Mapa del lago (para saber que celda es que)
LAKE_MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]
CELL_COLORS = {"S": "#4fc3f7", "F": "#e0f2f1", "H": "#ef9a9a", "G": "#a5d6a7"}
ARROW = {0: "←", 1: "↓", 2: "→", 3: "↑"}


def plot_value_and_policy(V, policy, title, ax_v, ax_p):
    #heatmap de valores y mapa de politica.
    n = 4
    V_grid = V.reshape(n, n)
    P_grid = policy.reshape(n, n)

    # funcion de valor
    im = ax_v.imshow(V_grid, cmap="YlGn", vmin=0, vmax=1)
    for r in range(n):
        for c in range(n):
            cell = LAKE_MAP[r][c]
            symbol = {"S": "S", "H": "H", "G": "G"}.get(cell, "")
            label = f"{V_grid[r, c]:.3f}\n{symbol}" if symbol else f"{V_grid[r, c]:.3f}"
            ax_v.text(c, r, label, ha="center", va="center", fontsize=9,
                      color="black" if V_grid[r, c] < 0.7 else "white")
    ax_v.set_title(f"Funcion de Valor - {title}", fontsize=11, fontweight="bold")
    ax_v.set_xticks(range(n)); ax_v.set_yticks(range(n))
    ax_v.set_xticklabels(range(n)); ax_v.set_yticklabels(range(n))
    plt.colorbar(im, ax=ax_v, fraction=0.046, pad=0.04)

    # Politica
    for r in range(n):
        for c in range(n):
            cell = LAKE_MAP[r][c]
            color = CELL_COLORS[cell]
            ax_p.add_patch(plt.Rectangle((c - 0.5, n - 1 - r - 0.5), 1, 1,
                                          color=color, zorder=0))
            if cell in ("H", "G"):
                ax_p.text(c, n - 1 - r, cell, ha="center", va="center",
                          fontsize=14, fontweight="bold",
                          color="#b71c1c" if cell == "H" else "#1b5e20")
            else:
                ax_p.text(c, n - 1 - r, ARROW[P_grid[r, c]],
                          ha="center", va="center", fontsize=18, color="#1a237e")
    ax_p.set_xlim(-0.5, n - 0.5); ax_p.set_ylim(-0.5, n - 0.5)
    ax_p.set_xticks(range(n)); ax_p.set_yticks(range(n))
    ax_p.set_xticklabels(range(n))
    ax_p.set_yticklabels(range(n - 1, -1, -1))
    ax_p.set_title(f"Politica Optima - {title}", fontsize=11, fontweight="bold")
    ax_p.set_aspect("equal")


# ─────────────────────────────────────────────
#  PARTE A: VALUE ITERATION
# ─────────────────────────────────────────────

def value_iteration(env, gamma=0.99, theta=1e-8):
    """
        env    : entorno de Gym con dinamica env.P disponible
        gamma  : factor de descuento
        theta  : criterio de convergencia (delta minimo)

        V      : funcion de valor optima (array de tamaño n_states)
        policy : politica optima (array de tamaño n_states)
    """
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)

    iteration = 0
    while True:
        delta = 0
        for s in range(n_states):
            v_prev = V[s]
            action_values = np.zeros(n_actions)
            for a in range(n_actions):
                for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                    action_values[a] += prob * (reward + gamma * V[next_state])
            V[s] = np.max(action_values)
            delta = max(delta, abs(v_prev - V[s]))

        iteration += 1
        
        if delta < theta:
            print(f"  Value Iteration convergió en {iteration} iteraciones (delta={delta:.2e})")
            break
        
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            for prob, next_state, reward, done in env.unwrapped.P[s][a]:
                action_values[a] += prob * (reward + gamma * V[next_state])
        policy[s] = np.argmax(action_values)

    return V, policy


def evaluate_policy(env, policy, n_episodes=1000):

    successes = 0
    total_reward = 0.0

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy[state]
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        total_reward += reward
        if reward > 0:
            successes += 1

    return successes / n_episodes, total_reward / n_episodes


def run_parte_a():
    #Value Iteration en modo deterministico y estocastico
    print("\n" + "="*60)
    print("  PARTE A – VALUE ITERATION (basado en modelo)")
    print("="*60)

    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Parte A – Value Iteration: Frozen Lake\n"
                 "(deterministico y estocastico)", fontsize=14, fontweight="bold", y=1.01)

    for i, (label, slippery) in enumerate([("Deterministico", False),
                                            ("Estocastico",   True)]):
        print(f"\n[{label}]")
        env = gym.make("FrozenLake-v1", is_slippery=slippery)

        V, policy = value_iteration(env, gamma=0.99, theta=1e-8)

        success_rate, avg_reward = evaluate_policy(env, policy, n_episodes=1000)
        print(f"  Tasa de exito : {success_rate*100:.1f}%")
        print(f"  Recompensa media: {avg_reward:.4f}")

        results[label] = {"V": V, "policy": policy,
                          "success_rate": success_rate,
                          "avg_reward": avg_reward}

        print(f"\n  Función de Valor (reshape 4×4):")
        print(np.round(V.reshape(4, 4), 4))
        print(f"\n  Política (0=← 1=↓ 2=→ 3=↑):")
        print(policy.reshape(4, 4))

        plot_value_and_policy(V, policy, label, axes[i][0], axes[i][1])
        env.close()

    print("\n[Comparacion – Parte A]")
    print(f"  {'Modo':<20} {'Tasa exito':>12} {'Recompensa media':>18}")
    print("  " + "-"*52)
    for label, r in results.items():
        print(f"  {label:<20} {r['success_rate']*100:>11.1f}% {r['avg_reward']:>18.4f}")

    plt.tight_layout()
    plt.savefig("./parte_a_value_iteration.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Grafico guardado: parte_a_value_iteration.png")
    return results


# ─────────────────────────────────────────────
#  PARTE B: Q-LEARNING
# ─────────────────────────────────────────────

def q_learning(env, episodes=10000, alpha=0.1, gamma=0.99,
               epsilon=1.0, epsilon_decay=0.999, epsilon_min=0.01):
    """
        epsilon: probabilidad inicial de exploracion
        epsilon_decay: decaimiento de epsilon por episodio
        epsilon_min: valor minimo de epsilon
        Q : tabla Q aprendida (states x actions)
        policy: politica derivada de Q (argmax por estado)
        rewards: recompensas acumuladas por episodio
    """
    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    # Inicializa la tabla Q con 0s
    Q = np.zeros((n_states, n_actions))
    rewards = []
    eps = epsilon

    for ep in range(episodes):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            # epsilon-greedy
            if np.random.random() < eps:
                action = env.action_space.sample() # exploracion
            else:
                action = np.argmax(Q[state]) # explotacion
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Bellman off-policy
            best_next = np.max(Q[next_state])
            Q[state, action] += alpha * (reward + gamma * best_next - Q[state, action])

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        # decaimiento de epsilon
        if eps > epsilon_min:
            eps *= epsilon_decay

    # politica final (argmax de Q por estado)
    policy = np.argmax(Q, axis=1)

    return Q, policy, rewards


def evaluate_q_policy(env, Q, n_episodes=1000):
    #desempeño con episodios de prueba usando la politica greedy Q
    successes = 0
    total_reward = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        total_reward += reward
        if reward > 0:
            successes += 1
    return successes / n_episodes, total_reward / n_episodes


def run_parte_b(vi_results):
    #Q-Learning en modo deterministico y estocastico
    print("\n" + "="*60)
    print("  PARTE B  Q-LEARNING (libre de modelo)")
    print("="*60)

    ql_results = {}
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Parte B Q-Learning: Frozen Lake\n"
                 "(deterministico y estocastico)", fontsize=14, fontweight="bold", y=1.01)

    WINDOW = 200   # ventana de suavizado para la curva de recompensas

    for i, (label, slippery) in enumerate([("Deterministico", False),
                                            ("Estocastico",   True)]):
        print(f"\n[{label}]")
        #  entrenar en cada modo
        env = gym.make("FrozenLake-v1", is_slippery=slippery)
        Q, policy, rewards = q_learning(env, episodes=15000, alpha=0.1,
                                        gamma=0.99, epsilon=1.0,
                                        epsilon_decay=0.9995, epsilon_min=0.01)

        success_rate, avg_reward = evaluate_q_policy(env, Q, n_episodes=1000)
        print(f"  Tasa de exito : {success_rate*100:.1f}%")
        print(f"  Recompensa media: {avg_reward:.4f}")

        ql_results[label] = {"Q": Q, "policy": policy, "rewards": rewards,
                             "success_rate": success_rate, "avg_reward": avg_reward}

        print(f"\n  Tabla Q (primeras 4 filas, acciones 0-3):")
        print(np.round(Q[:4], 4))
        print(f"\n  Politica (0=← 1=↓ 2=→ 3=↑):")
        print(policy.reshape(4, 4))

        # grafica de recompensas durante el entrenamiento
        ax_r = axes[i][0]
        smoothed = np.convolve(rewards, np.ones(WINDOW)/WINDOW, mode="valid")
        ax_r.plot(rewards, alpha=0.2, color="#90caf9", linewidth=0.6)
        ax_r.plot(range(WINDOW - 1, len(rewards)), smoothed,
                  color="#1565c0", linewidth=2, label=f"Media móvil ({WINDOW} ep.)")
        ax_r.set_title(f"Recompensas – {label}", fontsize=11, fontweight="bold")
        ax_r.set_xlabel("Episodio"); ax_r.set_ylabel("Recompensa")
        ax_r.legend(fontsize=8); ax_r.grid(True, alpha=0.3)

        # funcion de valor implicita en Q y politica
        V_from_Q = np.max(Q, axis=1)
        plot_value_and_policy(V_from_Q, policy, label, axes[i][1], axes[i][2])
        env.close()

    plt.tight_layout()
    plt.savefig("./parte_b_q_learning.png",
                dpi=150, bbox_inches="tight")
    plt.close()

    # compara con Value Iteration
    print("\n[Comparacion – Parte A vs Parte B]")
    print(f"  {'Modo':<20} {'Método':<18} {'Tasa éxito':>12} {'Recompensa media':>18}")
    print("  " + "-"*72)
    for label in ["Deterministico", "Estocastico"]:
        vi = vi_results[label]
        ql = ql_results[label]
        print(f"  {label:<20} {'Value Iteration':<18} {vi['success_rate']*100:>11.1f}% {vi['avg_reward']:>18.4f}")
        print(f"  {label:<20} {'Q-Learning':<18} {ql['success_rate']*100:>11.1f}% {ql['avg_reward']:>18.4f}")
        print()

    return ql_results


# ─────────────────────────────────────────────
#  GRAFICO DE COMPARACION FINAL
# ─────────────────────────────────────────────

def plot_comparison(vi_results, ql_results):
    labels   = ["Deterministico\nValue Iter.", "Deterministico\nQ-Learning",
                 "Estocastico\nValue Iter.", "Estocastico\nQ-Learning"]
    values   = [vi_results["Deterministico"]["success_rate"] * 100,
                ql_results["Deterministico"]["success_rate"] * 100,
                vi_results["Estocastico"]["success_rate"]    * 100,
                ql_results["Estocastico"]["success_rate"]    * 100]
    colors   = ["#1565c0", "#42a5f5", "#b71c1c", "#ef9a9a"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Tasa de exito (%)", fontsize=12)
    ax.set_title("Comparacion de metodos – Frozen Lake\nValue Iteration vs Q-Learning",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    vi_patch = mpatches.Patch(color="#1565c0", label="Value Iteration (det.)")
    ql_patch = mpatches.Patch(color="#42a5f5", label="Q-Learning (det.)")
    vi_s_patch = mpatches.Patch(color="#b71c1c", label="Value Iteration (esto.)")
    ql_s_patch = mpatches.Patch(color="#ef9a9a", label="Q-Learning (esto.)")
    ax.legend(handles=[vi_patch, ql_patch, vi_s_patch, ql_s_patch],
              loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig("./comparacion_final.png", dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    vi_results = run_parte_a()
    ql_results = run_parte_b(vi_results)
    print("\n" + "="*60)
    print("  GRAFICO COMPARATIVO FINAL")
    print("="*60)
    plot_comparison(vi_results, ql_results)
