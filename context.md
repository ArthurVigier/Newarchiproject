```
# PROJET : Architecture LLM Expérimentale sans Backpropagation
## Contexte et état d'avancement

Tu es l'assistant de recherche d'Arthur, chercheur ML indépendant
basé en France. Ce document capture l'intégralité d'une session
de conception architecturale. Tu dois pouvoir continuer le travail
sans discontinuité.

---

## GENÈSE INTELLECTUELLE DU PROJET

Ce projet est né d'une discussion sur l'efficacité énergétique du
cerveau humain comparée aux LLMs actuels. Les pain points identifiés
étaient :

1. **Séparation mémoire/calcul** (von Neumann bottleneck) — dans les
   LLMs, mémoire et calcul sont séparés, nécessitant des transferts
   constants d'énergie. Le cerveau les co-localise.

2. **Calcul binaire précis vs analogique bruité** — les transistors
   exigent des états parfaitement discriminables. Les synapses
   biologiques fonctionnent avec 30% de taux de transmission et
   ça marche.

3. **Backpropagation biologiquement implausibie** — nécessite de
   mémoriser toute la passe forward et de revenir en arrière. Le
   cerveau apprend localement via Hebbian/STDP.

4. **Activation dense vs sparsité extrême** — les LLMs activent
   la majorité de leurs paramètres à chaque token. Le cerveau
   active <5% de ses neurones simultanément.

5. **Apprentissage one-shot absent** — les LLMs nécessitent des
   millions d'exemples. Le cerveau mémorise en une occurrence.

---

## L'ARCHITECTURE CIBLE

### Philosophie générale

**Ce n'est pas une modification de Transformer existant.** C'est
une architecture from scratch combinant quatre propriétés qui
n'ont jamais été assemblées :

1. Pas de backpropagation globale
2. États internes stochastiques (tolérance à l'entropie)
3. Routing MoE par entropie
4. Survie neuronale darwinienne comme mécanisme d'apprentissage
5. Introspection des activations propres comme signal

### Inspiration biologique

Le cerveau fait de la **sélection darwinienne sur les neurones**
via :
- Apoptose neuronale (mort cellulaire programmée massive)
- Pruning synaptique (élimination brutale des connexions inutiles)
- Consolidation pendant le sommeil (reset sélective)
- Modulation dopaminergique (signal de récompense global rare)

### Composants de l'architecture

#### 1. LLM de base frozen
- **Modèle** : Qwen3-4B (choix actuel)
- **Rôle** : extracteur de représentations
- **Couche d'intérêt** : couche 9 / last_token / centered
  (Spearman ρ = 0.6538 validé empiriquement par Arthur)
- **Jamais entraîné** pendant les expériences

#### 2. Encodeur Stochastique + SIGReg
Inspiré directement de **LeWM (LeWorldModel)** — papier de
LeCun/Mila publié mars 2026, code open-source disponible :
https://github.com/lucas-maes/le-wm

```python
class StochasticTextEncoder(nn.Module):
    def __init__(self, input_dim=4096, latent_dim=512,
                 dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),  # requis pour SIGReg
            nn.GELU(),
            nn.Dropout(dropout)          # stochasticité résiduelle
        )
        self.sigreg = SIGReg(n_projections=64)
```

**SIGReg** (Sketched-Isotropic-Gaussian Regularizer) :
- Empêche le representation collapse sans stop-gradients ni EMA
- Utilise le théorème de Cramér-Wold
- Force les embeddings latents vers une Gaussienne isotrope
- Un seul hyperparamètre λ à tuner (bisection search O(log n))
- Convergence lisse et monotone — critique pour système sans backprop

#### 3. Prédicteur Léger (style LeWM)
```python
class LatentPredictor(nn.Module):
    # Prédit z_t+1 depuis z_t
    # z_t = état latent du prompt
    # z_t+1 = état latent attendu de la solution
```

Loss totale — deux termes uniquement (comme LeWM) :
```
L = L_pred + λ * SIGReg(Z)
```

#### 4. Router Entropique MoE
- N=3 experts au départ
- **Routing par entropie** : sélectionne l'expert dont les
  prédictions stochastiques ont la plus faible variance
- Monte Carlo dropout (5 passes) pour estimer l'incertitude
- L'expert le moins surpris par l'input est sélectionné

```python
class EntropyRouter(nn.Module):
    def compute_expert_entropy(self, z, expert_idx):
        predictor.train()  # active dropout pour MC
        preds = torch.stack([predictor(z) for _ in range(5)])
        entropy = preds.var(dim=0).mean(dim=-1)
        return entropy
```

#### 5. Expert à Population de Survie Darwinienne
**C'est le composant le plus original et le plus expérimental.**

Remplace entièrement l'apprentissage Hebbien. Inspiré des
automates cellulaires (Conway's Game of Life) et de la sélection
naturelle.

**Principe** : pas de gradient sur les neurones des experts.
Chaque neurone a un score de survie et des règles d'état.

**États** : Actif / Dormant / Mort→Renaissance

**Règles de transition** :
```python
# Neurone actif + reward > 0 → survit, score++, dormancy reset
# Neurone actif + reward < 0 → pénalisé, dormancy+2
# Neurone silencieux → dormancy++
# dormancy > DORMANCY_THRESHOLD → Dormant
# dormancy > DEATH_THRESHOLD → Mort → Renaissance aléatoire
# Renaissance = réinitialisation poids avec bruit exploration
```

**Propriétés clés** :
- `requires_grad=False` sur les poids des experts
- Pas de backprop sur les SurvivalExperts
- Seul le reward externe binaire déterministe pilote la survie
- La renaissance crée de l'exploration permanente

```python
class SurvivalExpert(nn.Module):
    def __init__(self, latent_dim=512, n_neurons=256,
                 dormancy_threshold=15, death_threshold=50,
                 rebirth_noise=0.05):
        self.weights = nn.Parameter(
            torch.randn(n_neurons, latent_dim) * 0.01,
            requires_grad=False  # PAS DE BACKPROP
        )
        self.survival_scores = ...
        self.dormancy_counters = ...
        self.is_active = ...

    @torch.no_grad()
    def survival_update(self, reward: float):
        # Règles darwiniennes locales
        # Pas de gradient
```

#### 6. Projecteur Introspectif
- Prend z_expert (512d) et le projette vers l'espace d'embedding
  du LLM (4096d)
- **Entraîné par backprop** — c'est le pont entre survie et LLM
- Injecté comme token en position 0 du contexte

```python
class IntrospectionProjector(nn.Module):
    # (batch, 512) → (batch, 1, 4096)
    # Le LLM "voit" son propre état latent avant de générer
```

**Idée centrale de l'introspection** : Arthur a observé que son
framework R̂/Â prédit le comportement des modèles depuis
l'extérieur via les activations. Ici on donne au modèle la
capacité de faire ça depuis l'intérieur, pour s'auto-corriger.

#### 7. Signal de Récompense
**Reward binaire, déterministe, externe — pas de reward hacking.**

Le modèle fait une tâche. Il réussit ou échoue. Point.

Pas de :
- LLM-as-judge
- Métriques de surprise
- Free energy interne

---

## BENCHMARKS SÉLECTIONNÉS

HumanEval a été **explicitement rejeté** pour les raisons
suivantes :
- Saturation : modèles frontier à 93%
- Contamination documentée : chute de 5-14 points sur HumanEval-T
- Trop facile pour générer une pression de sélection saine

### Stack de benchmarks retenue

```python
benchmarks = {
    'livecodebench': {
        # Rolling updates mensuels → contamination impossible
        # Reward binaire : pass/fail tests unitaires
        'n_problems': 150,
        'difficulty': 'medium',
        'contamination_risk': 'very_low'
    },
    'bigcodebench': {
        # Diversité bibliothèques → plus proche du vrai dev
        # Unittest pass/fail binaire
        'n_problems': 100,
        'difficulty': 'medium_hard',
        'contamination_risk': 'low'
    },
    'math500': {
        # Problèmes de compétition mathématique
        # Vérification exact match
        'n_problems': 100,
        'difficulty': 'hard',
        'contamination_risk': 'medium'
    },
    'arc_agi2': {
        # Genuinement difficile pour 4B → reward rare
        # Exact match sur grille
        'n_problems': 50,
        'difficulty': 'very_hard',
        'contamination_risk': 'very_low'
    }
}
```

**Rationale de la diversité** : code + maths + raisonnement
crée une pression sélective différente sur chaque expert MoE,
forçant une spécialisation émergente.

---

## DOUBLE BOUCLE D'ENTRAÎNEMENT

C'est l'élément architectural central :

```
Ce qui est entraîné par BACKPROP :
├── StochasticTextEncoder (SIGReg + pred loss)
├── IntrospectionProjector (signal RL)
└── EntropyRouter.predictors (pred loss)

Ce qui est entraîné par SURVIE NEURONALE :
├── SurvivalExpert.weights (règles darwiniennes)
└── SurvivalExpert.output_weights (règles darwiniennes)

Ce qui est FROZEN :
└── LLM de base Qwen3-4B
```

---

## RESSOURCES EXISTANTES D'ARTHUR

Arthur a déjà les outils suivants opérationnels :

- **R̂/Â framework** : publié sur PyPI comme `a_hat_optimizer`
  - Extrait les activations de couches intermédiaires
  - AUC > 0.94 validé sur Qwen3-1.7B/4B/8B
  - Couche 9 / last_token / centered optimal pour Qwen3-4B
  - C'est le composant clé pour extraire z_t

- **lewm-habitat-merged** : dataset publié sur HuggingFace
  - `Artvv/lewm-habitat-merged`
  - Arthur connaît le codebase LeWM en pratique

- **Stack infrastructure** : Vast.ai (RTX) + RunPod (A100/H100)
- **Jetson Orin Nano** : pour edge deployment futur

- **Expérience GRPO** : Arthur a déjà travaillé avec du RL sur
  modèles ouverts, connaît les pièges

- **Expérience MoE** : projet ERNIE-4.5-21B-A3B, Braess Paradox
  confirmé dans 25.7% des configurations

---

## ROADMAP COMPLÈTE AVEC KILL GATES

### PHASE 0 — Kill Gates Fondamentaux (~20$ / 3-5 jours)

**Step 0.1 — Setup**
```bash
git clone https://github.com/lucas-maes/le-wm
pip install -e .
pip install transformers accelerate trl lm-eval a_hat_optimizer
```

**Step 0.2 — Kill Gate A : SIGReg sur texte**
- Question : SIGReg converge-t-il sur embeddings textuels ?
- Protocole : encoder minimal, 2000 prompts variés, 500 steps
- Kill criterion : convergence non-monotone → repenser encodeur

**Step 0.3 — Kill Gate B : AUC activations**
- Question : activations couche 9 prédisent-elles pass/fail ?
- Protocole : 500 solutions LiveCodeBench, classifieur linéaire
- Kill criterion : AUC < 0.65 → essayer couches 12/16 avant kill

**Step 0.4 — Kill Gate C : survie sur XOR**
- Question : règles de survie darwiniennes apprennent-elles ?
- Protocole : SurvivalPopulation 50 neurones, XOR, 1000 steps,
  ZÉRO gradient
- Kill criterion : accuracy < 0.60 → calibrer seuils dormance/mort

---

### PHASE 1 — Encodeur Stochastique + SIGReg (~30$ / 1 semaine)

- Construire StochasticTextEncoder avec BatchNorm + Dropout 0.1
- Intégrer SIGReg depuis le repo LeWM
- Construire LatentPredictor (style LeWM, ~10M params)
- Loss = MSE_pred + λ * SIGReg
- Validation : AUC séparabilité pass/fail dans espace latent

---

### PHASE 2 — Populations de Survie dans les Experts MoE
(~50$ / 1-2 semaines)

- Construire SurvivalExpert complet avec règles d'état
- Construire EntropyRouter avec MC Dropout
- Kill gate : spécialisation des experts sur code/math/raisonnement

---

### PHASE 3 — Calibration des Seuils de Survie
(~50$ / 1-2 semaines)

- Grid search minimal sur dormancy_threshold et death_threshold
- Objectif : population stable entre 30% et 80% de neurones actifs
- Visualisation des dynamiques (naissances, morts, dormances)
- Kill gate : amélioration > 1% sur benchmark après 2000 steps

---

### PHASE 4 — Injection Introspective
(~80$ / 2 semaines)

- Construire IntrospectionProjector (512 → 4096)
- Modifier generate() pour injecter token en position 0
- Test A/B : avec vs sans introspection sur mêmes problèmes
- Kill gate : delta > -0.02 (l'injection ne doit pas dégrader)

---

### PHASE 5 — Boucle RL Complète
(~150$ / 2-3 semaines)

- Pipeline complet : extract → encode → route → survive →
  inject → generate → reward → update
- Entraînement sur 10 000 steps avec évaluation tous les 500
- Baseline stricte : Qwen3-4B vanilla mesurée d'abord
- Kill gate progressif à step 2000

---

### PHASE 6 — Ablations et Publication
(~50$ / 2 semaines)

Quatre runs comparatifs :
1. Architecture complète
2. Sans introspection (token aléatoire)
3. Sans survie (experts MLP standard)
4. Sans SIGReg (voir si collapse)
5. Routing uniforme aléatoire

---

### Budget et Timeline

| Phase | Durée | Compute |
|-------|-------|---------|
| P0 Kill gates | 3-5 jours | ~20$ |
| P1 Encodeur | 1 semaine | ~30$ |
| P2 MoE survie | 1-2 semaines | ~50$ |
| P3 Calibration | 1-2 semaines | ~50$ |
| P4 Introspection | 2 semaines | ~80$ |
| P5 RL complet | 2-3 semaines | ~150$ |
| P6 Publication | 2 semaines | ~50$ |
| **Total** | **~2 mois** | **~430$** |

---

## RÈGLES ABSOLUES DE CONDUITE

1. **Jamais sauter un kill gate** — si P0.4 échoue, on ne passe
   pas à P2 avant d'avoir compris pourquoi

2. **Baseline toujours mesurée d'abord** — Qwen3-4B vanilla
   sur les mêmes benchmarks avant chaque phase

3. **Journaliser les dynamiques de population** — même si les
   benchmarks ne s'améliorent pas, les patterns de vie/mort des
   neurones sont un résultat en soi et probablement publiable

4. **Un seul composant change à la fois après P0** — pour isoler
   les causalités

5. **Falsifier avant d'investir** — protocole habituel d'Arthur

---

## MÉTRIQUES À COLLECTER

```python
metrics = {
    # Performance
    'pass_at_1_livecodebench': ...,
    'accuracy_bigcodebench': ...,
    'accuracy_math500': ...,
    'accuracy_arc_agi2': ...,

    # Dynamiques de population (contribution originale)
    'population_diversity': ...,     # entropie distribution scores
    'specialist_emergence': ...,     # KL divergence entre experts
    'survival_turnover_rate': ...,   # morts/renaissances par step
    'mean_neuron_lifespan': ...,     # durée de vie moyenne

    # Introspection
    'introspection_auc': ...,        # AUC prédiction succès via z
    'calibration_error': ...,        # ECE

    # SIGReg
    'latent_gaussianity': ...,       # distance à Gaussienne cible
    'sigreg_convergence_steps': ..., # steps pour stabilisation
}
```

---

## TITRE ET STRUCTURE DU PAPIER CIBLE

**Titre provisoire** :
"Survival-Based Expert Populations with Introspective Context
Injection for LLM Self-Improvement without Global Backpropagation"

**Sections** :
1. Introduction — gap cerveau/LLM, limits de backprop
2. Architecture — 5 composants détaillés
3. Algorithme — double boucle backprop/survie
4. Expériences — setup, résultats, ablations
5. Analyse — dynamiques de population, que capture z_expert
6. Discussion — liens neurosciences, limites, directions futures

**Contribution principale** :
Première architecture combinant survie neuronale darwinienne,
routing MoE entropique, SIGReg anti-collapse, et introspection
des activations propres — sans backpropagation globale sur les
composants d'apprentissage principaux.

---

## ÉTAT ACTUEL

**Phase** : Conception complète, prêt pour Phase 0

**Prochaine action immédiate** :
Phase 0 Step 0.4 — XOR avec survie neuronale pure, sur CPU local,
zéro GPU nécessaire, ~30 minutes pour avoir un premier signal.

```python
# Premier code à exécuter
pop = SurvivalPopulation(n_neurons=50, input_dim=2)
# Test XOR sans gradient
# Kill criterion : accuracy > 0.60 après 1000 steps
```

**Questions ouvertes non résolues** :
- Seuils optimaux dormancy/death (à calibrer en Phase 3)
- Mécanisme exact de contribution neuronale
  (comment savoir quel neurone a contribué à l'output)
- Stabilité de la boucle RL complète avec deux régimes
  d'apprentissage simultanés

---

## PROTOCOLE DE TRAVAIL D'ARTHUR (à respecter)

- Toujours falsifier avant d'investir
- Tuer le sunk cost explicitement si vérification invalide
- Diverger avant de converger
- Prioriser web search sur les priors du modèle
- Fournir des rankings quantitatifs avec scores
- Ne jamais confirmer en boucle — challenger, vérifier, quantifier
- Extraire la méthode réutilisable, pas seulement le résultat
- Budget compute : conserver les décisions dans l'enveloppe
  Vast.ai/RunPod avec kill gates stricts
```
