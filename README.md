# Atelier Transformers Encoder-Decoder

**Le pont entre comprendre et générer**

Atelier Deep Learning (modélisation du langage, NLP) animé par **Lotfi Djebbar** et **Issam Menas**.

L'atelier explique comment fonctionne un Transformer encoder-decoder : l'architecture qui lit une séquence puis en écrit une autre (traduction, résumé, transcription). Il se déroule en trois temps :

1. **Une présentation** pour poser les notions.
2. **Un notebook pratique** où l'on ouvre un vrai modèle de traduction français → anglais.
3. **Le Transformer Grand Prix** : une course en direct où chaque étudiant résout des épreuves sur les notions vues, avec le classement affiché au projecteur.

## Contenu du dépôt

| Fichier | Pour qui | Description |
|---|---|---|
| [`Transformers_Encoder_Decoder.pptx`](Transformers_Encoder_Decoder.pptx) | Tout le monde | La présentation utilisée pendant l'atelier (24 slides) |
| [`Transformer_Encoder_Decoder.ipynb`](Transformer_Encoder_Decoder.ipynb) | Les étudiants | Le notebook pratique : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LotfiDjebbar/atelier-transformers/blob/main/Transformer_Encoder_Decoder.ipynb) |
| [`Transformer_Grand_Prix_Serveur.ipynb`](Transformer_Grand_Prix_Serveur.ipynb) | L'animateur | Le serveur de la course : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LotfiDjebbar/atelier-transformers/blob/main/Transformer_Grand_Prix_Serveur.ipynb) |

Rien à installer : un navigateur et un compte Google pour Colab suffisent.

## 1. La présentation

La présentation suit un fil simple : on connaît déjà l'encoder (famille BERT, qui comprend) et le decoder (famille GPT, qui génère). Que se passe-t-il si on branche les deux ?

**Introduction**
- D'où viennent les Transformers : l'article *Attention Is All You Need* (2017), conçu au départ pour la traduction automatique.
- Le problème à résoudre : une séquence entre, une autre sort, avec un ordre, une longueur ou une nature différente (« chat noir » devient « black cat », 3 pages deviennent 3 lignes, de l'audio devient du texte).

**Acte 1 : l'architecture**
- Vue d'ensemble : deux moitiés (encoder et decoder) reliées par un pont, la cross-attention.
- L'encoder lit toute la phrase source une seule fois et produit la « mémoire » : un vecteur par token.
- Rappel de l'attention avec Query, Key et Value : `Attention(Q, K, V) = softmax(Q·Kᵀ / √d) · V`.
- Les trois attentions et qui regarde qui : self-attention (encoder), masked self-attention (decoder), cross-attention (du decoder vers l'encoder).
- Le decoder et ses trois étages : relire ce qu'on a écrit, consulter la phrase source, digérer l'information.
- La cross-attention en détail : Q vient du decoder, K et V viennent de l'encoder.
- L'analogie étudiante : l'encoder-decoder, c'est l'examen à documents autorisés.

**Acte 2 : la génération pas à pas**
- On suit la traduction de « Le chat dort » en quatre étapes, du premier mot jusqu'au token de fin `</s>`.
- Trois stratégies pour choisir le mot suivant : greedy, beam search et sampling (avec la température).
- L'entraînement par teacher forcing : le decoder reçoit la vraie traduction décalée d'un cran, et le masque l'empêche de tricher.

**Acte 3 : la famille et la pratique**
- Les modèles à connaître : Transformer (2017), T5 et BART (2019), NLLB-200 et Whisper (2022).
- Quel Transformer pour quelle tâche : encoder seul, decoder seul ou encoder-decoder.
- À retenir : l'encoder lit une fois, le decoder écrit un token à la fois, la cross-attention relie les deux.

## 2. Le notebook pratique

Le notebook utilise **MarianMT** (`Helsinki-NLP/opus-mt-fr-en`), un vrai Transformer encoder-decoder entraîné pour traduire du français vers l'anglais. Chaque section reprend une slide de la présentation, avec cette fois les vraies valeurs du modèle.

| Section | Ce qu'on fait |
|---|---|
| 0. Installation | Installer `transformers`, `sentencepiece` et `sacremoses` |
| 1. Charger le modèle | Charger le tokenizer et le modèle, en gardant les poids d'attention |
| 2. Traduire en trois lignes | Tokenizer, `generate`, `decode` |
| 3. Ouvrir la boîte | Compter les couches de l'encoder et du decoder, retrouver les trois étages du decoder dans le code |
| 4. Les tokens | Voir comment une phrase est découpée en tokens et en IDs |
| 5. L'encoder : la mémoire | Faire tourner l'encoder seul et observer la forme `[1, nombre de tokens, 512]` |
| 6. Le decoder, token par token | Refaire `generate` à la main et afficher les 3 mots les plus probables à chaque étape |
| 7. La carte de cross-attention | Afficher quel mot français le decoder regarde pour écrire chaque mot anglais |
| 8. Trois stratégies | Comparer greedy, beam search et sampling à plusieurs températures |
| 9. L'application | Une petite interface pour traduire une phrase et afficher sa carte d'attention |

**Pour commencer :** cliquez sur le bouton *Open in Colab*, puis exécutez les cellules dans l'ordre avec **Shift + Entrée**. Le modèle tourne sur CPU ; un GPU est utilisé automatiquement s'il est disponible.

## 3. Le Transformer Grand Prix

Une course en direct qui sert d'évaluation ludique. Le notebook serveur, lancé par l'animateur dans Colab, crée un petit site web avec Flask et un lien public gratuit avec cloudflared :

| Page | Qui l'ouvre | Ce qu'on y voit |
|---|---|---|
| **Dashboard** | Le projecteur | Le circuit, les voitures qui avancent, le chrono, le classement en direct, les feux de départ et le podium |
| **Cockpit** (`/play`) | Chaque étudiant sur son PC | Le secteur à résoudre, son chrono, sa position et le **garage** : des outils qui font tourner le vrai modèle |

### Les 6 secteurs

| Secteur | Notion travaillée |
|---|---|
| 1. Le compte-tours | Encoder vs decoder : combien de fois chacun tourne |
| 2. La chicane des masques | Masked self-attention et cross-attention |
| 3. Le duel Q·K | Query et Key : produits scalaires à la main |
| 4. Le virage de la température | Softmax et sampling |
| 5. Le muret du beam search | Greedy vs beam search |
| 6. Drapeau à damier | Le pipeline complet de traduction |

### Règlement

| Règle | Effet |
|---|---|
| Mauvaise réponse | +15 s |
| Arrêt au stand (indice) | +30 s |
| Premier à boucler un secteur | Secteur violet au tableau |
| Classement | Le plus de secteurs, puis le meilleur temps pénalités comprises |

### Lancer la course (animateur)

1. Ouvrez le notebook serveur dans Colab et lancez **Exécution → Tout exécuter** (environ 2 minutes).
2. La dernière cellule affiche deux liens : le lien **projecteur**, qui contient le code animateur (à ne pas montrer), et le lien **pilotes**.
3. Ouvrez le lien projecteur en plein écran (F11) sur le PC branché au vidéoprojecteur. Il affiche le lien pilotes et un QR code.
4. Les étudiants ouvrent le lien ou scannent le QR code, choisissent un pseudo et une couleur : leur voiture apparaît sur la grille.
5. Cliquez sur **Départ** : les cinq feux rouges s'allument puis s'éteignent, et la course commence.
6. Quand tout le monde a franchi la ligne, ou quand vous cliquez sur **Fin de course**, le podium s'affiche.

Le lien public change à chaque lancement. Gardez l'onglet Colab ouvert pendant toute la course : s'il se ferme, le serveur s'arrête.

## Technologies

- [Hugging Face Transformers](https://huggingface.co/docs/transformers) et le modèle [MarianMT `opus-mt-fr-en`](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en)
- PyTorch, Matplotlib, ipywidgets
- Flask, segno (QR code) et cloudflared pour le serveur de course
- Google Colab
