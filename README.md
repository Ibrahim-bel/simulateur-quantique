# Simulateur Quantique Interactif ⚛️

Application web interactive en **Dash** permettant d'explorer et visualiser les concepts fondamentaux de l'informatique quantique : superposition, intrication, portes quantiques, sphère de Bloch, histogrammes de mesures…

Tout est centralisé dans un fichier Python unique qui contient l'application Dash, les visualisations Plotly et la logique mathématique basée sur NumPy.

---

## 🚀 Technologies utilisées

- **Dash** — interface web et callbacks interactifs  
- **Plotly** — visualisations 3D et graphiques  
- **NumPy** — calcul matriciel pour les états quantiques  
- **Gunicorn** — serveur de production pour Render  
- **Python 3.x**

---

## 📦 Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/Ibrahim-bel/simulateur-quantique.git
cd simulateur-quantique
```

### 2. Créer un environnement virtuel (recommandé)
```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

**Contenu de `requirements.txt` :**
```text
dash
plotly
numpy
gunicorn
```

---

## ▶️ Lancer le projet en local

### ✔️ Méthode : Lancer avec Python
```bash
python q.py
```

Ensuite ouvrir le navigateur sur :
```
http://127.0.0.1:8050
```

---

## ☁️ Déploiement sur Render

### Fichier Procfile

Deux options sont possibles :

#### 🟦 Option A — Lancement direct (simple)

**Contenu de `Procfile` :**
```Procfile
web: python q.py
```

➡️ Fonctionne, mais Render impose un port différent de 8050, d'où…

#### 🟩 Option B — Recommandé (Gunicorn)

**Contenu de `Procfile` :**
```Procfile
web: gunicorn q:server --bind 0.0.0.0:$PORT
```

- `q` = nom du fichier `q.py`
- `server` = objet Flask exposé par Dash
- `$PORT` = variable imposée par Render

### Étapes Render :

1. Pousser le projet sur GitHub
2. Aller sur **Render** → **New** → **Web Service**
3. Sélectionner le repo GitHub
4. Paramètres :
   - **Environment** : Python 3
   - **Build Command** :
```bash
     pip install -r requirements.txt
```
   - **Start Command** : laisser vide → Render utilisera automatiquement le `Procfile`

---

## 📁 Structure du projet
```
.
├── q.py             # Application Dash + simulateur quantique
├── requirements.txt # Dépendances (dash, plotly, numpy, gunicorn)
├── Procfile         # Configuration Render
└── README.md        # Ce fichier
```

---

## ✨ Fonctionnalités principales

### Interface Interactive

- **Trois onglets thématiques** :
  - 🌀 **Superposition** : Visualisation sur sphère de Bloch, application de portes quantiques (Hadamard, Pauli-X, Pauli-Z)
  - 🔗 **Intrication** : Création d'états de Bell (Hadamard + CNOT), diagramme de circuit, métrique d'intrication
  - 🏗️ **Architecture** : Exploration des 4 couches d'un ordinateur quantique (physique, contrôle, logique, logicielle)

### Simulateur Quantique

- **Classe `QuantumSimulator`** :
  - États quantiques sur 2 qubits (espace de Hilbert 4D)
  - Portes quantiques : Hadamard, CNOT, Pauli-X/Y/Z, Rotations Rx/Rz
  - Calcul des probabilités de mesure : P(i) = |αᵢ|²
  - Métrique d'intrication : Entropie de von Neumann S = -Tr(ρ log₂(ρ))
  - Simulation de mesures répétées (1000 shots)
  - Historique des opérations

### Visualisations Avancées

- **Sphère de Bloch 3D** interactive avec axes X, Y, Z colorés
- **Distribution de probabilité** des états |00⟩, |01⟩, |10⟩, |11⟩
- **Diagramme de circuit quantique** avec symboles H (Hadamard), ● (contrôle), ⊕ (CNOT)
- **Histogramme de mesure** (simulation Monte Carlo sur 1000 essais)
- **Graphiques animés** mis à jour en temps réel

### Aspects Pédagogiques

- Explications théoriques détaillées pour chaque concept
- Formules mathématiques affichées : |ψ⟩ = α|0⟩ + β|1⟩
- Interprétations physiques (superposition, intrication, mesure)
- Spécifications techniques réelles (T₁, T₂, fréquences, températures)
- Architecture stratifiée d'un ordinateur quantique complet
- Progression pédagogique : du qubit unique à l'intrication multi-qubits

### Détails Techniques

L'application utilise :
- **NumPy** pour les opérations d'algèbre linéaire (matrices unitaires 4×4)
- **Plotly Graph Objects** pour les visualisations 3D et 2D
- **Dash Callbacks** pour l'interactivité en temps réel
- **Architecture front-end/back-end** séparée proprement

**Code structure dans `q.py` :**
```python
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # Pour Gunicorn / Render

# Classe principale
class QuantumSimulator:
    def __init__(self, n_qubits=2):
        self.n_qubits = n_qubits
        self.dim = 2**n_qubits
        self.reset()
    
    def apply_hadamard(self, qubit): ...
    def apply_cnot(self, control, target): ...
    def calculate_entanglement(self): ...
    # etc.

# Visualisations
def create_bloch_sphere(simulator, qubit=0): ...
def create_state_visualization(simulator): ...
def create_circuit_diagram(operations): ...
def create_measurement_histogram(counts): ...

# Callbacks Dash pour l'interactivité
@app.callback(...)
def update_superposition_tab(...): ...

@app.callback(...)
def update_entanglement_tab(...): ...
```

---

## 🎓 Concepts Quantiques Implémentés

### 1. Superposition

**Principe** : Un qubit peut exister dans une combinaison linéaire de |0⟩ et |1⟩ :
```
|ψ⟩ = α|0⟩ + β|1⟩  avec  |α|² + |β|² = 1
```

**Implémentation** :
- Porte Hadamard : |0⟩ → (|0⟩ + |1⟩)/√2
- Visualisation sur sphère de Bloch
- Affichage des amplitudes complexes et probabilités

### 2. Intrication

**Principe** : Corrélation quantique entre qubits. État de Bell :
```
|Φ⁺⟩ = (|00⟩ + |11⟩)/√2
```

**Implémentation** :
- Création par Hadamard + CNOT
- Métrique S = -Tr(ρ_A log₂(ρ_A)) avec S=1 pour intrication maximale
- Histogramme montrant uniquement |00⟩ et |11⟩ (jamais |01⟩ ou |10⟩)

### 3. Architecture Complète

**4 couches d'un ordinateur quantique** :

1. **Couche Physique** : Qubits supraconducteurs à 15 mK, jonctions Josephson
2. **Couche Contrôle** : Signaux micro-ondes 4-8 GHz, AWG, FPGA
3. **Couche Logique** : Portes natives, compilation, correction d'erreurs
4. **Couche Logicielle** : Qiskit, algorithmes (Shor, Grover, VQE)

---

## 📌 Licence

MIT License

---

## 🤝 Contribution

Les PR et suggestions sont les bienvenues !

1. Fork du dépôt
2. `git checkout -b feature/nouvelle-fonction`
3. `git commit -m "Ajout nouvelle fonction"`
4. `git push origin feature/nouvelle-fonction`
5. Ouvrir une Pull Request

---

## 📚 Ressources Additionnelles

- [Documentation Qiskit](https://qiskit.org/documentation/)
- [IBM Quantum Experience](https://quantum-computing.ibm.com/)
- [Plotly Dash Documentation](https://dash.plotly.com/)
- [NumPy Documentation](https://numpy.org/doc/)

---

## 🔬 Auteur

Développé dans le cadre d'un projet pédagogique sur l'informatique quantique.

Pour toute question : [GitHub Issues](https://github.com/Ibrahim-bel/simulateur-quantique/issues)

---

## 🌐 Démo en ligne

Application déployée sur Render : [Lien vers l'application](https://votre-app.onrender.com)

*(Remplacer par l'URL réelle après déploiement)*

---

**Version** : 1.0.0  
**Dernière mise à jour** : 2025
