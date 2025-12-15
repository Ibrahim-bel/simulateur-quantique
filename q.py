import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
import dash

# ==========================================
# SIMULATION QUANTIQUE COMPLÈTE
# ==========================================

class QuantumSimulator:
    """Simulateur quantique éducatif pour comprendre les états et opérations"""
    
    def __init__(self, n_qubits=2):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        self.reset()
        
    def reset(self):
        """Réinitialise à l'état |0...0⟩"""
        self.state = np.zeros(self.dim, dtype=complex)
        self.state[0] = 1.0
        self.history = [self.state.copy()]
        self.operations = ["État initial |00⟩"]
        
    def get_state_vector(self):
        """Retourne le vecteur d'état actuel"""
        return self.state.copy()
    
    def get_probabilities(self):
        """Retourne les probabilités de mesure"""
        return np.abs(self.state)**2
    
    def apply_hadamard(self, qubit):
        """Applique une porte Hadamard sur un qubit"""
        H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        self._apply_single_qubit_gate(H, qubit)
        self.operations.append(f"Hadamard sur qubit {qubit}")
        self.history.append(self.state.copy())
        
    def apply_cnot(self, control, target):
        """Applique une porte CNOT"""
        gate = np.eye(self.dim, dtype=complex)
        
        for i in range(self.dim):
            # Vérifier si le bit de contrôle est à 1
            if (i >> (self.n_qubits - 1 - control)) & 1:
                # Inverser le bit cible
                j = i ^ (1 << (self.n_qubits - 1 - target))
                gate[i, i] = 0
                gate[i, j] = 1
                
        self.state = gate @ self.state
        self.operations.append(f"CNOT (contrôle={control}, cible={target})")
        self.history.append(self.state.copy())
        
    def apply_pauli_x(self, qubit):
        """Applique une porte Pauli-X (NOT quantique)"""
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        self._apply_single_qubit_gate(X, qubit)
        self.operations.append(f"Pauli-X sur qubit {qubit}")
        self.history.append(self.state.copy())
        
    def apply_pauli_z(self, qubit):
        """Applique une porte Pauli-Z (phase flip)"""
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        self._apply_single_qubit_gate(Z, qubit)
        self.operations.append(f"Pauli-Z sur qubit {qubit}")
        self.history.append(self.state.copy())
        
    def apply_rotation_x(self, qubit, theta):
        """Rotation autour de l'axe X"""
        Rx = np.array([
            [np.cos(theta/2), -1j*np.sin(theta/2)],
            [-1j*np.sin(theta/2), np.cos(theta/2)]
        ], dtype=complex)
        self._apply_single_qubit_gate(Rx, qubit)
        self.operations.append(f"Rx({theta:.2f}) sur qubit {qubit}")
        self.history.append(self.state.copy())
        
    def apply_rotation_z(self, qubit, theta):
        """Rotation autour de l'axe Z"""
        Rz = np.array([
            [np.exp(-1j*theta/2), 0],
            [0, np.exp(1j*theta/2)]
        ], dtype=complex)
        self._apply_single_qubit_gate(Rz, qubit)
        self.operations.append(f"Rz({theta:.2f}) sur qubit {qubit}")
        self.history.append(self.state.copy())
        
    def _apply_single_qubit_gate(self, gate, qubit):
        """Applique une porte à un qubit dans un système multi-qubits"""
        if self.n_qubits == 1:
            self.state = gate @ self.state
            return
            
        # Construction de la porte complète par produit tensoriel
        if qubit == 0:
            full_gate = gate
            for i in range(1, self.n_qubits):
                full_gate = np.kron(full_gate, np.eye(2))
        else:
            full_gate = np.eye(2)
            for i in range(1, self.n_qubits):
                if i == qubit:
                    full_gate = np.kron(full_gate, gate)
                else:
                    full_gate = np.kron(full_gate, np.eye(2))
                    
        self.state = full_gate @ self.state
        
    def measure(self, shots=1000):
        """Simule des mesures répétées"""
        probs = self.get_probabilities()
        outcomes = np.random.choice(self.dim, size=shots, p=probs)
        counts = {}
        for outcome in outcomes:
            binary = format(outcome, f'0{self.n_qubits}b')
            counts[binary] = counts.get(binary, 0) + 1
        return counts
    
    def get_bloch_coordinates(self, qubit=0):
        """Calcule les coordonnées de Bloch pour un qubit (trace partiel correct)"""
        # Matrice densité du système complet
        rho_full = np.outer(self.state, np.conj(self.state))
        
        if self.n_qubits == 1:
            rho = rho_full
        else:
            # Trace partiel pour obtenir la matrice densité réduite du qubit spécifié
            # Réorganiser le vecteur d'état en tenseur
            if qubit == 0:
                # Pour le qubit 0, on trace sur tous les autres qubits
                state_reshaped = self.state.reshape(2, 2**(self.n_qubits-1))
                rho = state_reshaped @ state_reshaped.conj().T
            else:
                # Pour le qubit 1 (dans un système 2-qubits)
                state_reshaped = self.state.reshape(2, 2)
                rho = np.zeros((2, 2), dtype=complex)
                for i in range(2):
                    rho += np.outer(state_reshaped[i, :], state_reshaped[i, :].conj())
            
        # Coordonnées de Bloch: (x, y, z) = (Tr(ρσx), Tr(ρσy), Tr(ρσz))
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        x = np.real(np.trace(rho @ sigma_x))
        y = np.real(np.trace(rho @ sigma_y))
        z = np.real(np.trace(rho @ sigma_z))
        
        return x, y, z
    
    def calculate_entanglement(self):
        """Calcule l'entropie d'intrication (mesure de l'intrication)"""
        if self.n_qubits != 2:
            return 0
        
        # Matrice densité réduite du premier qubit (trace partiel correct)
        state_matrix = self.state.reshape(2, 2)
        rho_A = state_matrix @ state_matrix.conj().T
        
        # Valeurs propres
        eigenvalues = np.linalg.eigvalsh(rho_A)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Éviter log(0)
        
        # Entropie de von Neumann
        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-15))
        return entropy


# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

def format_complex(z, precision=3):
    """Formate un nombre complexe de manière lisible"""
    real_part = z.real
    imag_part = z.imag
    
    # Si essentiellement réel
    if abs(imag_part) < 10**(-precision):
        return f'{real_part:.{precision}f}'
    
    # Si essentiellement imaginaire
    if abs(real_part) < 10**(-precision):
        if imag_part >= 0:
            return f'{imag_part:.{precision}f}i'
        else:
            return f'{imag_part:.{precision}f}i'
    
    # Cas général
    if imag_part >= 0:
        return f'{real_part:.{precision}f}+{imag_part:.{precision}f}i'
    else:
        return f'{real_part:.{precision}f}{imag_part:.{precision}f}i'


# ==========================================
# VISUALISATIONS AVANCÉES
# ==========================================

def create_state_visualization(simulator):
    """Crée une visualisation 3D de l'espace de Hilbert"""
    states = [format(i, f'0{simulator.n_qubits}b') for i in range(simulator.dim)]
    amplitudes = simulator.get_state_vector()
    probabilities = simulator.get_probabilities()
    
    # Graphique en barres montrant les probabilités
    fig = go.Figure()
    
    # Barres pour les probabilités
    fig.add_trace(go.Bar(
        x=states,
        y=probabilities,
        name='Probabilités',
        marker=dict(
            color=probabilities,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Probabilité")
        ),
        text=[f'{p:.3f}' for p in probabilities],
        textposition='outside',
        hovertemplate='<b>État |%{x}⟩</b><br>Probabilité: %{y:.4f}<br>Amplitude: ' + 
                      '<br>'.join([format_complex(amp) for amp in amplitudes]) + '<extra></extra>',
    ))
    
    fig.update_layout(
        title="Distribution de Probabilité des États Quantiques",
        xaxis_title="États de base",
        yaxis_title="Probabilité de mesure",
        template="plotly_dark",
        height=400,
        showlegend=False,
    )
    
    return fig

def create_bloch_sphere(simulator, qubit=0):
    """Crée une sphère de Bloch interactive"""
    try:
        x, y, z = simulator.get_bloch_coordinates(qubit)
    except:
        x, y, z = 0, 0, 1
    
    # Sphère
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig = go.Figure()
    
    # Surface de la sphère
    fig.add_trace(go.Surface(
        x=xs, y=ys, z=zs,
        opacity=0.3,
        colorscale=[[0, 'lightblue'], [1, 'lightblue']],
        showscale=False,
        hoverinfo='skip',
    ))
    
    # Axes
    axis_length = 1.3
    for axis, color, name in [
        ([axis_length, 0, 0], 'red', 'X'),
        ([0, axis_length, 0], 'green', 'Y'),
        ([0, 0, axis_length], 'blue', 'Z')
    ]:
        fig.add_trace(go.Scatter3d(
            x=[0, axis[0]], y=[0, axis[1]], z=[0, axis[2]],
            mode='lines+text',
            line=dict(color=color, width=4),
            text=['', name],
            textposition='top center',
            textfont=dict(size=16, color=color),
            showlegend=False,
            hoverinfo='skip',
        ))
    
    # Vecteur d'état
    fig.add_trace(go.Scatter3d(
        x=[0, x], y=[0, y], z=[0, z],
        mode='lines+markers',
        line=dict(color='yellow', width=6),
        marker=dict(size=[0, 12], color='yellow'),
        name='État |ψ⟩',
        hovertemplate=f'État: ({x:.3f}, {y:.3f}, {z:.3f})<extra></extra>',
    ))
    
    # Labels pour |0⟩ et |1⟩
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[1.5],
        mode='text',
        text=['|0⟩'],
        textfont=dict(size=18, color='white'),
        showlegend=False,
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[-1.5],
        mode='text',
        text=['|1⟩'],
        textfont=dict(size=18, color='white'),
        showlegend=False,
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-1.5, 1.5]),
            yaxis=dict(visible=False, range=[-1.5, 1.5]),
            zaxis=dict(visible=False, range=[-1.5, 1.5]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
            aspectmode='cube',
        ),
        title=f"Sphère de Bloch - Qubit {qubit}",
        template="plotly_dark",
        height=500,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    
    return fig

def create_circuit_diagram(operations):
    """Crée un diagramme du circuit quantique"""
    fig = go.Figure()
    
    # Lignes de qubits
    for i in range(2):
        fig.add_trace(go.Scatter(
            x=[0, len(operations)],
            y=[i, i],
            mode='lines',
            line=dict(color='white', width=2),
            showlegend=False,
            hoverinfo='skip',
        ))
    
    # Ajouter les portes
    gate_positions = []
    for idx, op in enumerate(operations[1:], 1):  # Skip "État initial"
        if 'Hadamard' in op:
            qubit = int(op.split()[-1])
            gate_positions.append((idx, qubit, 'H', 'cyan'))
        elif 'CNOT' in op:
            parts = op.split('contrôle=')[1].split(', cible=')
            control = int(parts[0])
            target = int(parts[1].rstrip(')'))
            gate_positions.append((idx, control, '●', 'orange'))
            gate_positions.append((idx, target, '⊕', 'orange'))
            # Ligne de connexion
            fig.add_trace(go.Scatter(
                x=[idx, idx],
                y=[control, target],
                mode='lines',
                line=dict(color='orange', width=3),
                showlegend=False,
                hoverinfo='skip',
            ))
        elif 'Pauli-X' in op:
            qubit = int(op.split()[-1])
            gate_positions.append((idx, qubit, 'X', 'red'))
        elif 'Pauli-Z' in op:
            qubit = int(op.split()[-1])
            gate_positions.append((idx, qubit, 'Z', 'purple'))
        elif 'Rx' in op:
            qubit = int(op.split()[-1])
            gate_positions.append((idx, qubit, 'Rx', 'magenta'))
        elif 'Rz' in op:
            qubit = int(op.split()[-1])
            gate_positions.append((idx, qubit, 'Rz', 'lime'))
    
    # Dessiner les portes
    for x, y, symbol, color in gate_positions:
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=30, color=color, line=dict(width=2, color='white')),
            text=symbol,
            textfont=dict(size=16, color='white'),
            showlegend=False,
            hovertext=f"Porte {symbol}",
        ))
    
    fig.update_layout(
        title="Diagramme du Circuit Quantique",
        xaxis=dict(title="Temps / Profondeur", showgrid=False, zeroline=False),
        yaxis=dict(
            title="Qubits",
            ticktext=['Qubit 1', 'Qubit 0'],
            tickvals=[1, 0],
            showgrid=False,
            zeroline=False,
        ),
        template="plotly_dark",
        height=300,
        showlegend=False,
    )
    
    return fig

def create_measurement_histogram(counts):
    """Crée un histogramme des résultats de mesure"""
    if not counts:
        # Retourner un graphique vide si pas de mesures
        fig = go.Figure()
        fig.update_layout(
            title="Résultats de Mesure (1000 shots)",
            template="plotly_dark",
            height=400,
        )
        return fig
    
    states = list(counts.keys())
    values = list(counts.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=states,
            y=values,
            marker=dict(
                color=values,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(title="Nombre de<br>mesures")
            ),
            text=values,
            textposition='outside',
        )
    ])
    
    fig.update_layout(
        title="Résultats de Mesure (1000 shots)",
        xaxis_title="États mesurés",
        yaxis_title="Nombre d'occurrences",
        template="plotly_dark",
        height=400,
    )
    
    return fig


# ==========================================
# APPLICATION DASH
# ==========================================

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Créer le simulateur global
sim = QuantumSimulator(n_qubits=2)

app.layout = html.Div(
    style={
        'backgroundColor': '#0f172a',
        'minHeight': '100vh',
        'padding': '20px',
        'fontFamily': 'Arial, sans-serif',
    },
    children=[
        dcc.Store(id='simulator-state', data={'step': 0}),
        
        # En-tête
        html.Div(
            style={
                'textAlign': 'center',
                'marginBottom': '40px',
                'padding': '30px',
                'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                'borderRadius': '15px',
                'boxShadow': '0 10px 40px rgba(0,0,0,0.3)',
            },
            children=[
                html.H1(
                    '⚛️ Fonctionnement Interne d\'un Ordinateur Quantique',
                    style={'color': 'white', 'marginBottom': '10px', 'fontSize': '36px'}
                ),
                html.P(
                    'Exploration interactive approfondie des principes quantiques',
                    style={'color': '#e0e7ff', 'fontSize': '18px', 'margin': '0'}
                ),
            ]
        ),
        
        # Onglets principaux
        dcc.Tabs(
            id='main-tabs',
            value='superposition',
            style={'marginBottom': '30px'},
            children=[
                # ===== ONGLET 1: SUPERPOSITION =====
                dcc.Tab(
                    label='🌀 Superposition',
                    value='superposition',
                    style={'backgroundColor': '#1e293b', 'color': '#94a3b8'},
                    selected_style={'backgroundColor': '#3b82f6', 'color': 'white'},
                    children=[
                        html.Div(style={'padding': '20px'}, children=[
                            # Explication théorique
                            html.Div(
                                style={
                                    'backgroundColor': '#1e293b',
                                    'padding': '25px',
                                    'borderRadius': '12px',
                                    'marginBottom': '30px',
                                    'border': '2px solid #3b82f6',
                                },
                                children=[
                                    html.H2('📚 Principe de Superposition', style={'color': '#60a5fa', 'marginBottom': '15px'}),
                                    html.P(
                                        "La superposition est un principe fondamental de la mécanique quantique. "
                                        "Contrairement à un bit classique qui est soit 0 soit 1, un qubit peut exister "
                                        "dans une combinaison linéaire de |0⟩ et |1⟩ simultanément.",
                                        style={'color': '#e2e8f0', 'lineHeight': '1.8', 'fontSize': '16px'}
                                    ),
                                    html.Div(
                                        style={
                                            'backgroundColor': '#0f172a',
                                            'padding': '20px',
                                            'borderRadius': '8px',
                                            'marginTop': '15px',
                                            'fontFamily': 'monospace',
                                        },
                                        children=[
                                            html.P('|ψ⟩ = α|0⟩ + β|1⟩', style={'color': '#22d3ee', 'fontSize': '20px', 'margin': '0'}),
                                            html.P('où |α|² + |β|² = 1', style={'color': '#94a3b8', 'fontSize': '14px', 'marginTop': '10px'}),
                                        ]
                                    ),
                                    html.P(
                                        "La porte Hadamard (H) est l'opération fondamentale pour créer une superposition. "
                                        "Elle transforme |0⟩ en une superposition égale: (|0⟩ + |1⟩)/√2",
                                        style={'color': '#e2e8f0', 'lineHeight': '1.8', 'fontSize': '16px', 'marginTop': '15px'}
                                    ),
                                ]
                            ),
                            
                            # Contrôles interactifs
                            html.Div(
                                style={
                                    'backgroundColor': '#1e293b',
                                    'padding': '25px',
                                    'borderRadius': '12px',
                                    'marginBottom': '30px',
                                },
                                children=[
                                    html.H3('🎛️ Contrôles Interactifs', style={'color': '#60a5fa', 'marginBottom': '20px'}),
                                    
                                    html.Div(
                                        style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px', 'marginBottom': '20px'},
                                        children=[
                                            html.Button(
                                                '🔄 Réinitialiser',
                                                id='btn-reset',
                                                n_clicks=0,
                                                style={
                                                    'padding': '12px 24px',
                                                    'fontSize': '16px',
                                                    'backgroundColor': '#ef4444',
                                                    'color': 'white',
                                                    'border': 'none',
                                                    'borderRadius': '8px',
                                                    'cursor': 'pointer',
                                                    'fontWeight': 'bold',
                                                }
                                            ),
                                            html.Button(
                                                '🌀 Appliquer Hadamard (Q0)',
                                                id='btn-hadamard-q0',
                                                n_clicks=0,
                                                style={
                                                    'padding': '12px 24px',
                                                    'fontSize': '16px',
                                                    'backgroundColor': '#3b82f6',
                                                    'color': 'white',
                                                    'border': 'none',
                                                    'borderRadius': '8px',
                                                    'cursor': 'pointer',
                                                    'fontWeight': 'bold',
                                                }
                                            ),
                                        ]
                                    ),
                                    
                                    html.Div(
                                        style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px'},
                                        children=[
                                            html.Button(
                                                '↔️ Pauli-X (NOT)',
                                                id='btn-pauli-x',
                                                n_clicks=0,
                                                style={
                                                    'padding': '12px 24px',
                                                    'fontSize': '16px',
                                                    'backgroundColor': '#8b5cf6',
                                                    'color': 'white',
                                                    'border': 'none',
                                                    'borderRadius': '8px',
                                                    'cursor': 'pointer',
                                                    'fontWeight': 'bold',
                                                }
                                            ),
                                            html.Button(
                                                '🎭 Pauli-Z (Phase)',
                                                id='btn-pauli-z',
                                                n_clicks=0,
                                                style={
                                                    'padding': '12px 24px',
                                                    'fontSize': '16px',
                                                    'backgroundColor': '#ec4899',
                                                    'color': 'white',
                                                    'border': 'none',
                                                    'borderRadius': '8px',
                                                    'cursor': 'pointer',
                                                    'fontWeight': 'bold',
                                                }
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            
                            # Visualisations
                            html.Div(
                                style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'},
                                children=[
                                    html.Div([
                                        dcc.Graph(id='bloch-sphere-q0', config={'displayModeBar': False}),
                                    ]),
                                    html.Div([
                                        dcc.Graph(id='state-distribution', config={'displayModeBar': False}),
                                    ]),
                                ]
                            ),
                            
                            # Informations sur l'état
                            html.Div(
                                id='state-info-super',
                                style={
                                    'backgroundColor': '#1e293b',
                                    'padding': '25px',
                                    'borderRadius': '12px',
                                    'marginTop': '20px',
                                    'border': '2px solid #10b981',
                                }
                            ),
                        ])
                    ]
                ),
                
                # ===== ONGLET 2: INTRICATION =====
                dcc.Tab(
                    label='🔗 Intrication',
                    value='entanglement',
                    style={'backgroundColor': '#1e293b', 'color': '#94a3b8'},
                    selected_style={'backgroundColor': '#8b5cf6', 'color': 'white'},
                    children=[
                        html.Div(style={'padding': '20px'}, children=[
                            # Explication théorique
                            html.Div(
                                style={
                                    'backgroundColor': '#1e293b',
                                    'padding': '25px',
                                    'borderRadius': '12px',
                                    'marginBottom': '30px',
                                    'border': '2px solid #8b5cf6',
                                },
                                children=[
                                    html.H2('📚 Intrication Quantique', style={'color': '#a78bfa', 'marginBottom': '15px'}),
                                    html.P(
                                        "L'intrication est une corrélation quantique entre deux ou plusieurs qubits. "
                                        "Lorsque des qubits sont intriqués, l'état d'un qubit dépend instantanément de l'état de l'autre, "
                                        "quelle que soit la distance qui les sépare. Einstein appelait cela 'action fantôme à distance'.",
                                        style={'color': '#e2e8f0', 'lineHeight': '1.8', 'fontSize': '16px'}
                                    ),
                                    html.Div(
                                        style={
                                            'backgroundColor': '#0f172a',
                                            'padding': '20px',
                                            'borderRadius': '8px',
                                            'marginTop': '15px',
                                            'fontFamily': 'monospace',
                                        },
                                        children=[
                                            html.P('État de Bell: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2', style={'color': '#c084fc', 'fontSize': '20px', 'margin': '0'}),
                                            html.P('Créé par: H ⊗ I puis CNOT', style={'color': '#94a3b8', 'fontSize': '14px', 'marginTop': '10px'}),
                                        ]
                                    ),
                                    html.P(
                                        "La porte CNOT (Controlled-NOT) est la clé pour créer l'intrication. "
                                        "Elle inverse le qubit cible si et seulement si le qubit contrôle est dans l'état |1⟩.",
                                        style={'color': '#e2e8f0', 'lineHeight': '1.8', 'fontSize': '16px', 'marginTop': '15px'}
                                    ),
                                ]
                            ),
                            
                            # Contrôles
                            html.Div(
                                style={
                                    'backgroundColor': '#1e293b',
                                    'padding': '25px',
                                    'borderRadius': '12px',
                                    'marginBottom': '30px',
                                },
                                children=[
                                    html.H3('🎛️ Création d\'un État de Bell', style={'color': '#a78bfa', 'marginBottom': '20px'}),
                                    
                                    html.Div(
                                        style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr', 'gap': '15px'},
                                        children=[
                                            html.Button(
                                                '1️⃣ Réinitialiser',
                                                id='btn-bell-reset',
                                                n_clicks=0,
                                                style={
                                                    'padding': '12px 24px',
                                                    'fontSize': '16px',
                                                    'backgroundColor': '#ef4444',
                                                    'color': 'white',
                                                    'border': 'none',
                                                    'borderRadius': '8px',
                                                    'cursor': 'pointer',
                                                    'fontWeight': 'bold',
                                                }
                                            ),
                                            html.Button(
                                                '2️⃣ Hadamard sur Q0',
                                                id='btn-bell-h',
                                                n_clicks=0,
                                                style={
                                                    'padding': '12px 24px',
                                                    'fontSize': '16px',
                                                    'backgroundColor': '#3b82f6',
                                                    'color': 'white',
                                                    'border': 'none',
                                                    'borderRadius': '8px',
                                                    'cursor': 'pointer',
                                                    'fontWeight': 'bold',
                                                }
                                            ),
                                            html.Button(
                                                '3️⃣ CNOT (Q0→Q1)',
                                                id='btn-cnot',
                                                n_clicks=0,
                                                style={
                                                    'padding': '12px 24px',
                                                    'fontSize': '16px',
                                                    'backgroundColor': '#8b5cf6',
                                                    'color': 'white',
                                                    'border': 'none',
                                                    'borderRadius': '8px',
                                                    'cursor': 'pointer',
                                                    'fontWeight': 'bold',
                                                }
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            
                            # Diagramme du circuit
                            html.Div([
                                dcc.Graph(id='circuit-diagram', config={'displayModeBar': False}),
                            ]),
                            
                            # Visualisations
                            html.Div(
                                style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px', 'marginTop': '20px'},
                                children=[
                                    html.Div([
                                        dcc.Graph(id='entanglement-state-dist', config={'displayModeBar': False}),
                                    ]),
                                    html.Div([
                                        dcc.Graph(id='measurement-histogram', config={'displayModeBar': False}),
                                    ]),
                                ]
                            ),
                            
                            # Métrique d'intrication
                            html.Div(
                                id='entanglement-metric',
                                style={
                                    'backgroundColor': '#1e293b',
                                    'padding': '25px',
                                    'borderRadius': '12px',
                                    'marginTop': '20px',
                                    'border': '2px solid #a78bfa',
                                }
                            ),
                        ])
                    ]
                ),
                
                # ===== ONGLET 3: ARCHITECTURE =====
                dcc.Tab(
                    label='🏗️ Architecture Quantique',
                    value='architecture',
                    style={'backgroundColor': '#1e293b', 'color': '#94a3b8'},
                    selected_style={'backgroundColor': '#10b981', 'color': 'white'},
                    children=[
                        html.Div(style={'padding': '20px'}, children=[
                            html.Div(
                                style={
                                    'backgroundColor': '#1e293b',
                                    'padding': '25px',
                                    'borderRadius': '12px',
                                    'marginBottom': '30px',
                                },
                                children=[
                                    html.H2('🏗️ Architecture Complète d\'un Ordinateur Quantique', 
                                           style={'color': '#34d399', 'marginBottom': '20px'}),
                                    
                                    # Couches d'architecture
                                    html.Div(
                                        style={'display': 'flex', 'flexDirection': 'column', 'gap': '15px'},
                                        children=[
                                            # Couche 4: Logicielle
                                            html.Div(
                                                style={
                                                    'backgroundColor': '#a855f7',
                                                    'padding': '20px',
                                                    'borderRadius': '10px',
                                                    'color': 'white',
                                                },
                                                children=[
                                                    html.H3('4️⃣ Couche Logicielle', style={'margin': '0 0 10px 0'}),
                                                    html.P('• Qiskit, Cirq, PyQuil - Frameworks de programmation', 
                                                          style={'margin': '5px 0'}),
                                                    html.P('• Compilation de circuits en portes natives', 
                                                          style={'margin': '5px 0'}),
                                                    html.P('• Algorithmes quantiques (Shor, Grover, VQE)', 
                                                          style={'margin': '5px 0'}),
                                                    html.P('• Interface utilisateur et API cloud', 
                                                          style={'margin': '5px 0'}),
                                                ]
                                            ),
                                            
                                            html.Div(
                                                style={'textAlign': 'center', 'color': '#94a3b8'},
                                                children=['⬇️ Translation en instructions matérielles']
                                            ),
                                            
                                            # Couche 3: Logique
                                            html.Div(
                                                style={
                                                    'backgroundColor': '#3b82f6',
                                                    'padding': '20px',
                                                    'borderRadius': '10px',
                                                    'color': 'white',
                                                },
                                                children=[
                                                    html.H3('3️⃣ Couche Logique Quantique', style={'margin': '0 0 10px 0'}),
                                                    html.P('• Portes quantiques universelles (H, CNOT, T, S)', 
                                                          style={'margin': '5px 0'}),
                                                    html.P('• Optimisation de profondeur de circuit', 
                                                          style={'margin': '5px 0'}),
                                                    html.P('• Mapping de qubits logiques → physiques', 
                                                          style={'margin': '5px 0'}),
                                                    html.P('• Correction d\'erreurs quantiques (QEC)', 
                                                          style={'margin': '5px 0'}),
                                                ]
                                            ),
                                            
                                            html.Div(
                                                style={'textAlign': 'center', 'color': '#94a3b8'},
                                                children=['⬇️ Génération de séquences de pulses']
                                            ),
                                            
                                            # Couche 2: Contrôle
                                            html.Div(
                                                style={
                                                    'backgroundColor': '#06b6d4',
                                                    'padding': '20px',
                                                    'borderRadius': '10px',
                                                    'color': 'white',
                                                },
                                                children=[
                                                    html.H3('2️⃣ Couche de Contrôle Électronique', style={'margin': '0 0 10px 0'}),
                                                    html.P('• AWG (Générateurs de formes d\'onde arbitraires)', 
                                                          style={'margin': '5px 0'}),
                                                    html.P('• Signaux micro-ondes (4-8 GHz) pour manipuler les qubits', 
                                                          style={'margin': '5px 0'}),
                                                    html.P('• Système de mesure dispersive (réflectométrie RF)', 
                                                          style={'margin': '5px 0'}),
                                                    html.P('• FPGA pour feedback en temps réel (<1 μs)', 
                                                          style={'margin': '5px 0'}),
                                                ]
                                            ),
                                            
                                            html.Div(
                                                style={'textAlign': 'center', 'color': '#94a3b8'},
                                                children=['⬇️ Transmission par câbles cryogéniques']
                                            ),
                                            
                                            # Couche 1: Physique
                                            html.Div(
                                                style={
                                                    'backgroundColor': '#10b981',
                                                    'padding': '20px',
                                                    'borderRadius': '10px',
                                                    'color': 'white',
                                                },
                                                children=[
                                                    html.H3('1️⃣ Couche Physique (Matériel)', style={'margin': '0 0 10px 0'}),
                                                    html.P('• Qubits supraconducteurs (Transmons, Flux qubits)', 
                                                          style={'margin': '5px 0'}),
                                                    html.P('• Réfrigérateur à dilution (10-15 mK)', 
                                                          style={'margin': '5px 0'}),
                                                    html.P('• Blindage électromagnétique multicouche', 
                                                          style={'margin': '5px 0'}),
                                                    html.P('• Puce quantique gravée sur silicium/saphir', 
                                                          style={'margin': '5px 0'}),
                                                ]
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            
                            # Détails techniques
                            html.Div(
                                style={
                                    'display': 'grid',
                                    'gridTemplateColumns': '1fr 1fr',
                                    'gap': '20px',
                                    'marginTop': '30px',
                                },
                                children=[
                                    html.Div(
                                        style={
                                            'backgroundColor': '#1e293b',
                                            'padding': '20px',
                                            'borderRadius': '12px',
                                            'border': '2px solid #3b82f6',
                                        },
                                        children=[
                                            html.H3('🔬 Qubits Supraconducteurs', style={'color': '#60a5fa', 'marginBottom': '15px'}),
                                            html.P('• Type: Transmon (circuit LC avec jonction Josephson)', 
                                                  style={'color': '#e2e8f0', 'margin': '8px 0'}),
                                            html.P('• Fréquence: 4-8 GHz', 
                                                  style={'color': '#e2e8f0', 'margin': '8px 0'}),
                                            html.P('• Anharmonicité: ~300 MHz', 
                                                  style={'color': '#e2e8f0', 'margin': '8px 0'}),
                                            html.P('• T₁ (relaxation): 50-200 μs', 
                                                  style={'color': '#e2e8f0', 'margin': '8px 0'}),
                                            html.P('• T₂ (cohérence): 20-150 μs', 
                                                  style={'color': '#e2e8f0', 'margin': '8px 0'}),
                                            html.P('• Durée porte 1-qubit: 20-50 ns', 
                                                  style={'color': '#e2e8f0', 'margin': '8px 0'}),
                                            html.P('• Durée porte 2-qubits: 100-500 ns', 
                                                  style={'color': '#e2e8f0', 'margin': '8px 0'}),
                                        ]
                                    ),
                                    
                                    html.Div(
                                        style={
                                            'backgroundColor': '#1e293b',
                                            'padding': '20px',
                                            'borderRadius': '12px',
                                            'border': '2px solid #06b6d4',
                                        },
                                        children=[
                                            html.H3('🌡️ Infrastructure Cryogénique', style={'color': '#22d3ee', 'marginBottom': '15px'}),
                                            html.P('• Température base: 10-15 mK', 
                                                  style={'color': '#e2e8f0', 'margin': '8px 0'}),
                                            html.P('• Puissance de refroidissement: ~10 μW à 100 mK', 
                                                  style={'color': '#e2e8f0', 'margin': '8px 0'}),
                                            html.P('• Temps de refroidissement: 24-48 heures', 
                                                  style={'color': '#e2e8f0', 'margin': '8px 0'}),
                                            html.P('• Étages de température: 6-7 (300K → 15mK)', 
                                                  style={'color': '#e2e8f0', 'margin': '8px 0'}),
                                            html.P('• Coût: 500k-2M€ par système', 
                                                  style={'color': '#e2e8f0', 'margin': '8px 0'}),
                                            html.P('• Isolation vibratoire: critique pour la cohérence', 
                                                  style={'color': '#e2e8f0', 'margin': '8px 0'}),
                                        ]
                                    ),
                                ]
                            ),
                        ])
                    ]
                ),
            ]
        ),
        
        # Footer
        html.Div(
            style={
                'textAlign': 'center',
                'marginTop': '40px',
                'padding': '20px',
                'color': '#64748b',
                'borderTop': '1px solid #334155',
            },
            children=[
                'Simulateur quantique éducatif | Basé sur les principes de Qiskit et IBM Quantum | IQ01 | BELAYACHI Ibrahim, FADOUACHE Kenza, MOUAYED Dina'
            ]
        ),
    ]
)


# ==========================================
# CALLBACKS
# ==========================================

@app.callback(
    [Output('bloch-sphere-q0', 'figure'),
     Output('state-distribution', 'figure'),
     Output('state-info-super', 'children')],
    [Input('btn-reset', 'n_clicks'),
     Input('btn-hadamard-q0', 'n_clicks'),
     Input('btn-pauli-x', 'n_clicks'),
     Input('btn-pauli-z', 'n_clicks')],
    prevent_initial_call=False
)
def update_superposition_tab(n_reset, n_h, n_x, n_z):
    ctx = dash.callback_context
    
    if not ctx.triggered or ctx.triggered[0]['prop_id'] == 'btn-reset.n_clicks':
        sim.reset()
    elif ctx.triggered[0]['prop_id'] == 'btn-hadamard-q0.n_clicks' and n_h > 0:
        sim.apply_hadamard(0)
    elif ctx.triggered[0]['prop_id'] == 'btn-pauli-x.n_clicks' and n_x > 0:
        sim.apply_pauli_x(0)  # Applique X sur qubit 0
    elif ctx.triggered[0]['prop_id'] == 'btn-pauli-z.n_clicks' and n_z > 0:
        sim.apply_pauli_z(0)  # Applique Z sur qubit 0
    
    # Créer les visualisations
    bloch_fig = create_bloch_sphere(sim, qubit=0)
    dist_fig = create_state_visualization(sim)
    
    # Informations sur l'état
    state_vector = sim.get_state_vector()
    probs = sim.get_probabilities()
    
    # Calculer les coordonnées de Bloch pour afficher l'effet
    x, y, z = sim.get_bloch_coordinates(0)
    
    info_div = html.Div([
        html.H3('📊 Informations sur l\'État Quantique', style={'color': '#34d399', 'marginBottom': '15px'}),
        html.Div(
            style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px'},
            children=[
                html.Div([
                    html.H4('Vecteur d\'État (avec phases complexes)', style={'color': '#60a5fa', 'fontSize': '16px'}),
                    html.Div(
                        style={'fontFamily': 'monospace', 'color': '#e2e8f0', 'fontSize': '14px'},
                        children=[
                            html.P(f'|ψ⟩ = ({format_complex(state_vector[0])})|00⟩ +'),
                            html.P(f'      ({format_complex(state_vector[1])})|01⟩ +'),
                            html.P(f'      ({format_complex(state_vector[2])})|10⟩ +'),
                            html.P(f'      ({format_complex(state_vector[3])})|11⟩'),
                        ]
                    ),
                    html.Div(
                        style={'marginTop': '10px', 'padding': '10px', 'backgroundColor': '#0f172a', 'borderRadius': '5px'},
                        children=[
                            html.P('Coordonnées Bloch (Qubit 0):', style={'color': '#22d3ee', 'fontSize': '12px', 'margin': '0 0 5px 0'}),
                            html.P(f'x={x:.3f}, y={y:.3f}, z={z:.3f}', style={'color': '#fbbf24', 'fontSize': '12px', 'margin': '0', 'fontFamily': 'monospace'}),
                        ]
                    ),
                ]),
                html.Div([
                    html.H4('Probabilités de Mesure', style={'color': '#a78bfa', 'fontSize': '16px'}),
                    html.Div(
                        style={'fontFamily': 'monospace', 'color': '#e2e8f0', 'fontSize': '14px'},
                        children=[
                            html.P(f'P(|00⟩) = {probs[0]:.4f} = {probs[0]*100:.2f}%'),
                            html.P(f'P(|01⟩) = {probs[1]:.4f} = {probs[1]*100:.2f}%'),
                            html.P(f'P(|10⟩) = {probs[2]:.4f} = {probs[2]*100:.2f}%'),
                            html.P(f'P(|11⟩) = {probs[3]:.4f} = {probs[3]*100:.2f}%'),
                        ]
                    ),
                ]),
            ]
        ),
        html.Div(
            style={'marginTop': '15px', 'padding': '15px', 'backgroundColor': '#0f172a', 'borderRadius': '8px'},
            children=[
                html.P(f'🔄 Nombre d\'opérations appliquées: {len(sim.operations) - 1}', 
                      style={'color': '#22d3ee', 'margin': '5px 0'}),
                html.P(f'📜 Dernière opération: {sim.operations[-1]}', 
                      style={'color': '#a78bfa', 'margin': '5px 0'}),
                html.P(f'✅ Normalisation: Σ|ψᵢ|² = {np.sum(probs):.6f} (doit être = 1.0)', 
                      style={'color': '#10b981', 'margin': '5px 0'}),
            ]
        ),
        # Explication des effets des portes
        html.Div(
            style={'marginTop': '15px', 'padding': '15px', 'backgroundColor': '#1e293b', 'borderRadius': '8px', 'border': '2px solid #f59e0b'},
            children=[
                html.H4('💡 Effet des Portes sur l\'État Actuel', style={'color': '#fbbf24', 'fontSize': '16px', 'marginBottom': '10px'}),
                html.Div(
                    style={'color': '#e2e8f0', 'fontSize': '14px', 'lineHeight': '1.6'},
                    children=[
                        html.P('🌀 Hadamard: Crée une superposition égale (|0⟩ + |1⟩)/√2'),
                        html.P('↔️ Pauli-X: Inverse |0⟩ ↔ |1⟩ (équivalent au NOT classique)'),
                        html.P('🎭 Pauli-Z: Ajoute une phase -1 à |1⟩ (|0⟩ reste, |1⟩ → -|1⟩)'),
                        html.P('📌 Conseil: Appliquez d\'abord Hadamard pour voir les effets de X et Z sur une superposition!', 
                              style={'color': '#22d3ee', 'fontWeight': 'bold', 'marginTop': '10px'}),
                    ]
                ),
            ]
        ),
        # Détection de l'état dominant
        html.Div(
            style={'marginTop': '15px', 'padding': '15px', 'backgroundColor': '#064e3b', 'borderRadius': '8px', 'border': '2px solid #10b981'},
            children=[
                html.H4('🎯 État Dominant Détecté', style={'color': '#34d399', 'fontSize': '16px', 'marginBottom': '10px'}),
                html.Div(
                    style={'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center', 'color': '#22d3ee', 'fontFamily': 'monospace'},
                    children=[
                        f"|{format(np.argmax(probs), '02b')}⟩ avec {np.max(probs)*100:.1f}% de probabilité"
                    ]
                ),
                html.P(
                    f"Distribution: |00⟩={probs[0]*100:.0f}% | |01⟩={probs[1]*100:.0f}% | |10⟩={probs[2]*100:.0f}% | |11⟩={probs[3]*100:.0f}%",
                    style={'color': '#a7f3d0', 'fontSize': '12px', 'textAlign': 'center', 'marginTop': '10px', 'fontFamily': 'monospace'}
                ),
            ]
        ),
    ])
    
    return bloch_fig, dist_fig, info_div


@app.callback(
    [Output('circuit-diagram', 'figure'),
     Output('entanglement-state-dist', 'figure'),
     Output('measurement-histogram', 'figure'),
     Output('entanglement-metric', 'children')],
    [Input('btn-bell-reset', 'n_clicks'),
     Input('btn-bell-h', 'n_clicks'),
     Input('btn-cnot', 'n_clicks')],
    prevent_initial_call=False
)
def update_entanglement_tab(n_reset, n_h, n_cnot):
    ctx = dash.callback_context
    
    if not ctx.triggered or ctx.triggered[0]['prop_id'] == 'btn-bell-reset.n_clicks':
        sim.reset()
    elif ctx.triggered[0]['prop_id'] == 'btn-bell-h.n_clicks' and n_h > 0:
        sim.apply_hadamard(0)
    elif ctx.triggered[0]['prop_id'] == 'btn-cnot.n_clicks' and n_cnot > 0:
        sim.apply_cnot(0, 1)
    
    # Créer les visualisations
    circuit_fig = create_circuit_diagram(sim.operations)
    dist_fig = create_state_visualization(sim)
    
    # Mesures
    counts = sim.measure(shots=1000)
    hist_fig = create_measurement_histogram(counts)
    
    # Métrique d'intrication
    entanglement = sim.calculate_entanglement()
    
    metric_div = html.Div([
        html.H3('🔗 Mesure d\'Intrication', style={'color': '#a78bfa', 'marginBottom': '15px'}),
        html.Div(
            style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px'},
            children=[
                html.Div([
                    html.H4('Entropie d\'Intrication', style={'color': '#c084fc', 'fontSize': '16px'}),
                    html.Div(
                        style={
                            'fontSize': '48px',
                            'fontWeight': 'bold',
                            'color': '#22d3ee' if entanglement > 0.9 else '#f59e0b' if entanglement > 0.5 else '#10b981',
                            'textAlign': 'center',
                            'marginTop': '10px',
                        },
                        children=[f'{entanglement:.4f}']
                    ),
                    html.P(
                        'Maximum = 1.0 (intrication maximale)',
                        style={'color': '#94a3b8', 'fontSize': '14px', 'textAlign': 'center', 'marginTop': '10px'}
                    ),
                ]),
                html.Div([
                    html.H4('Interprétation', style={'color': '#c084fc', 'fontSize': '16px'}),
                    html.Div(
                        style={'color': '#e2e8f0', 'fontSize': '14px', 'lineHeight': '1.8'},
                        children=[
                            html.P('• 0.0: Aucune intrication (état séparable)'),
                            html.P('• 0.5: Intrication partielle'),
                            html.P('• 1.0: Intrication maximale (état de Bell)'),
                            html.P(
                                f'État actuel: {"✅ État de Bell!" if entanglement > 0.99 else "🔄 En construction" if entanglement > 0.01 else "❌ Non intriqué"}',
                                style={'fontWeight': 'bold', 'marginTop': '10px'}
                            ),
                        ]
                    ),
                ]),
            ]
        ),
        html.Div(
            style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#0f172a', 'borderRadius': '8px'},
            children=[
                html.H4('🧮 Explication Mathématique', style={'color': '#22d3ee', 'fontSize': '16px', 'marginBottom': '10px'}),
                html.P(
                    "L'entropie d'intrication S est calculée via la matrice densité réduite ρ_A du premier qubit:",
                    style={'color': '#e2e8f0', 'fontSize': '14px', 'marginBottom': '8px'}
                ),
                html.Div(
                    style={'fontFamily': 'monospace', 'color': '#a78bfa', 'fontSize': '14px', 'marginLeft': '20px'},
                    children=[
                        html.P('S = -Tr(ρ_A log₂(ρ_A))'),
                        html.P('où ρ_A = Tr_B(|ψ⟩⟨ψ|)'),
                    ]
                ),
                html.P(
                    "Pour un état de Bell pur, S = 1.0. Pour un état séparable, S = 0.",
                    style={'color': '#e2e8f0', 'fontSize': '14px', 'marginTop': '10px'}
                ),
            ]
        ),
    ])
    
    return circuit_fig, dist_fig, hist_fig, metric_div


if __name__ == '__main__':
    print("=" * 70)
    print("🚀 SIMULATEUR QUANTIQUE INTERACTIF - VERSION CORRIGÉE")
    print("=" * 70)
    print("\n✅ Corrections appliquées:")
    print("  • Calcul de la sphère de Bloch avec trace partiel correct")
    print("  • Affichage des amplitudes complexes (phases visibles)")
    print("  • Gestion améliorée des nombres complexes")
    print("  • Vérification de normalisation ajoutée")
    print("\n🌐 Lancement du serveur Dash sur http://localhost:8050")
    print("=" * 70)
    print()
    
    app.run(debug=True, port=8050)
