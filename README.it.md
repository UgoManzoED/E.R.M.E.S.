*[Read this in English](README.md) | [Leggi in Italiano](README.it.md)*

# E.R.M.E.S. - Facial Expression Recognition System

**Università di Salerno** **Corso:** Machine Learning - A.A. 2025/2026  
**Professore:** Giuseppe POLESE, Loredana CARUCCIO

**Team:** * Ugo Manzo (Matricola: 0512119071) - [GitHub](https://github.com/UgoManzoED)
* Renato Natale (Matricola: 0512119641) - [GitHub](https://github.com/Re-1234)

> Progetto di ricerca e sviluppo nell'ambito della Computer Vision per la classificazione automatica delle espressioni facciali tramite Reti Neurali Convoluzionali (CNN).

![Project Status](https://img.shields.io/badge/Status-Active-success)
![License](https://img.shields.io/badge/License-APACHE-blue)

---

## Indice
* [Informazioni sul Progetto](#informazioni-sul-progetto)
* [Funzionalità e Metodologia](#funzionalità-e-metodologia)
* [Tecnologie](#tecnologie)
* [Per Iniziare](#per-iniziare)
  * [Prerequisiti](#prerequisiti)
  * [Installazione](#installazione)
* [Utilizzo](#utilizzo)
* [Benchmarking e Risultati](#benchmarking-e-risultati)
* [Sviluppi Futuri e Limiti del Modello](#sviluppi-futuri-e-limiti-del-modello)
* [Disclaimer e Limiti Etici](#disclaimer-e-limiti-etici)
* [Riferimenti Bibliografici](#riferimenti-bibliografici)

---

## Informazioni sul Progetto
**E.R.M.E.S.** è un progetto ingegneristico finalizzato all'implementazione di un'architettura neurale convoluzionale capace di estrarre e mappare i tratti somatici per decodificare lo stato emotivo umano. Il progetto adotta un approccio incrementale: parte misurando i limiti del Machine Learning classico (vettoriale), evolve verso una CNN proprietaria (E.R.M.E.S. V1), valida l'interpretabilità con l'**Explainable AI (Grad-CAM)** e testa i limiti del dominio tramite **Transfer Learning (VGG16)**.



L'addestramento e il benchmarking sono stati condotti sul dataset pubblico **FER-2013** (immagini 48x48 pixel in scala di grigi).

## Funzionalità e Metodologia
* **Architettura CNN Custom (E.R.M.E.S. V1):** Una rete da ~5.87 milioni di parametri progettata per elaborare la geometria spaziale 2D, dominando le baselines classiche con un'Accuracy del 62.00%.
* **Setup Sperimentale:** La rete è stata addestrata utilizzando l'ottimizzatore **Adam** per minimizzare la funzione di costo **Categorical Crossentropy** per la classificazione multi-classe: $\mathcal{L}=-\sum_{i=1}^{C}y_i\log(\hat{y}_i)$
* **Explainable AI (Trasparenza Decisionale):** Implementazione dell'algoritmo Grad-CAM per dimostrare che la rete estrae pattern anatomicamente coerenti (es. ancoraggio sulla glabella per le classi *Angry* e *Disgust*) ed è strutturalmente immune agli artefatti di background.
* **Pipeline Dati Ottimizzata:** Una pipeline asincrona basata su `tf.data` con *Data Augmentation* spaziale e iniezione di pesi dinamici per mitigare il severo sbilanciamento delle classi (16:1) e il rumore strutturale.
* **Benchmarking Completo:** Analisi comparativa che spazia dal Flattening con SVM/Random Forest fino all'Upsampling tensoriale con VGG16, dimostrando che il limite predittivo attuale risiede nella qualità del dato originale.

## Tecnologie
* [Python](https://www.python.org/)
* [TensorFlow / Keras](https://www.tensorflow.org/) (per architettura CNN, `tf.data` e TensorBoard)
* [Scikit-Learn](https://scikit-learn.org/) (per PCA, SVM, Random Forest)
* [Jupyter Notebook](https://jupyter.org/)

---

## Per Iniziare
Segui queste istruzioni per ottenere una copia locale e farla girare sulla tua macchina.

### Prerequisiti
È necessario avere Python installato sul proprio sistema. Si consiglia caldamente l'uso di un ambiente virtuale isolato.

### Installazione

1. Clona la repository:
```bash
git clone https://github.com/UgoManzoED/E.R.M.E.S..git
```

2. Crea e attiva l'ambiente virtuale:
```bash
python -m venv venv
# Su Windows
venv\Scripts\activate
# Su macOS/Linux
source venv/bin/activate
```

3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

4. Configura il dataset:
   * Scarica manualmente il [dataset FER-2013](https://www.kaggle.com/datasets/msambare/fer2013).
   * Posizionalo in una cartella denominata `data/` nella root del progetto.

---

## Utilizzo
Il progetto è suddiviso in Jupyter Notebooks sequenziali. Puoi eseguirli in ordine per riprodurre le fasi di esplorazione dati, addestramento e valutazione.

```text
ERMES-Project/
 |-- notebooks/
 |   |-- 01_EDA_Data_Analysis.ipynb        # Analisi statistica e anomalie
 |   |-- 02_Training_CNN.ipynb             # Ingestione tf.data, augmentation e class weights
 |   |-- 03_Model_Baseline.ipynb           # Flattening, PCA, SVM e Random Forest
 |   |-- 04_Model_CNN.ipynb                # Architettura E.R.M.E.S. V1 e training
 |   |-- 05_Explainable_AI_GradCAM.ipynb   # Trasparenza predittiva tramite mappe di calore
 |   |-- 06_Transfer_Learning_VGG16.ipynb  # Upsampling e fine-tuning VGG16
```

*Nota: I file dei pesi (`*.h5`), le directory dei log di TensorBoard e il dataset originale non sono versionati per motivi di spazio e per conformità alle best practices.*

## Benchmarking e Risultati

| Modello / Architettura | Accuracy Globale (%) | Macro F1-Score |
| :--- | :---: | :---: |
| Zero Rule Baseline (Classe Maggioritaria) | - | 0.06 |
| Random Baseline (Stratificata) | - | 0.14 |
| SVM Lineare + PCA | 34.00 | 0.28 |
| Random Forest (Ensemble Non-Lineare) | 46.00 | 0.44 |
| **E.R.M.E.S. V1 (CNN Custom)** | **62.00** | **0.60** |
| VGG16 (Upsampling + Fine-Tuning) | 62.00 | 0.61 |
| *Human Baseline (Upper Bound Teorico)* | *~65.00±5* | *-* |

## Sviluppi Futuri e Limiti del Modello
- [ ] **Bias Demografici:** Mitigare i bias di rappresentazione demografica intrinsecamente presenti nel dataset FER-2013.
- [ ] **Robustezza:** Migliorare la resilienza del modello in condizioni di scarsa illuminazione o in presenza di occlusioni fisiologiche.
- [ ] **Explainability:** Espandere l'analisi di Explainable AI integrando ulteriori metriche di attribuzione (es. SHAP, Integrated Gradients).

## Disclaimer e Limiti Etici
Il modello E.R.M.E.S. è concepito esclusivamente come prototipo di ricerca accademica in ambito informatico e **non possiede alcuna validità diagnostica o clinica**. Le performance del sistema sono intrinsecamente soggette ai bias di rappresentazione demografica presenti nei dati di addestramento e decadono sensibilmente in condizioni ambientali non ottimali.

## Riferimenti Bibliografici
1. **FER-2013:** Goodfellow, I.J., et al. (2013). *Challenges in Representation Learning: A report on three machine learning contests*.
2. **VGG16:** Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*.
3. **Grad-CAM:** Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*.
