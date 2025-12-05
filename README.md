# Multimodal ADR Normalization System (RxFusion)

A comprehensive research project exploring multimodal learning for **Adverse Drug Reaction (ADR) normalization** using both **user-generated text** and **user-generated images**. This work evaluates supervised encoder-based models, multimodal fusion strategies, and modern multimodal LLMs to answer how visual context improves clinical concept disambiguation.

This repository contains the complete codebase including data loaders, model architectures, fusion modules, training pipelines, evaluation utilities, and analysis tools used in our study.

---

# 1.  Overview and Motivation

Medical Entity Normalization (MEN) converts noisy symptom descriptions into standardized clinical concepts (e.g., MedDRA). Traditionally, MEN relies only on **text**, but real-world ADR reporting is shifting toward **multimodal** posts on platforms such as:

- Patient forums  
- Social media  
- Mobile health tracking apps  

Patients increasingly upload **photos** of rashes, bumps, redness, and lesions alongside textual descriptions. However:

- Text-only systems miss visual clues.  
- Image-only classifiers lack context (e.g., cause, duration, sensation).  
- No major research attempts **true multimodal ADR normalization**.

This project fills that gap by evaluating:

- Whether **text + images** outperform unimodal models  
- Which **fusion strategy** works best  
- Whether **multimodal LLMs** surpass supervised models  
- How **fairness** varies across skin types  

---

# 2. Research Questions

1. **RQ1:** Does combining user-generated photos with text improve ADR normalization accuracy compared to text-only or image-only systems?  
2. **RQ2:** How does visual context affect model interpretability, robustness, and fairness across diverse skin tones and image qualities?  
3. **RQ3:** Which vision-language architecture (LLaMA-Vision or SkinGPT) yields better multimodal ADR classification performance?  
4. **RQ4:** Among early fusion, late fusion, and gated fusion, which strategy produces the most stable and accurate multimodal alignment?

---

# 3. Dataset Selection and Rationale

True multimodal ADR datasets (paired text + image from patients) **do not exist publicly** due to privacy restrictions. To create a realistic research environment, we selected and synthesized datasets that approximate user-generated ADR data.

## 3.1 Textual Datasets (ADR Mentions)

### **1. CADEC Corpus**
- Forum posts from patients discussing medication experiences  
- Includes gold-standard MedDRA annotations  
- Rich, descriptive language → ideal for text-based normalization  

### **2. SMM4H ADR Dataset**
- Short, noisy Twitter posts mentioning side effects  
- Adds real-world variability, slang, emojis, abbreviations  
- Helps simulate mobile/social ADR reporting  

**Why chosen?**  
These represent the two linguistic extremes: structured + descriptive (CADEC) vs noisy + real-world (SMM4H). Together, they provide a full spectrum of ADR language.

---

## 3.2 Image Datasets (Dermatology Photos)

### **1. SkinCAP**
- ~4,000 image–caption pairs  
- 178 dermatology conditions  
- Expert-written captions  
- Perfect for **supervised multimodal experiments**  

### **2. DermNet**
- Thousands of clinical dermatology images  
- Diverse conditions resembling ADR manifestations  
- Good for broadening visual coverage  

### **3. ISIC Archive**
- High-quality lesion imagery  
- Useful for conditions similar to ADR rash-like patterns  

### **4. MMADE / DDI Dataset**
- Biopsy-confirmed images  
- Includes Fitzpatrick skin tone metadata → critical for fairness analysis  

**Why chosen?**  
Dermatologic ADRs often resemble eczema, urticaria, rash, redness, swelling, etc. These datasets provide realistic representations of such manifestations while supporting fairness evaluations.

---

## 3.3 Proxy Multimodal Dataset Construction

Since no true text–image ADR dataset exists, we constructed a **proxy multimodal dataset**:

- Extract ADR mentions from CADEC/SMM4H  
- Map them to relevant dermatology categories  
- Pair with visually aligned examples from SkinCAP / DermNet / ISIC  
- Validate pairs through keyword + ontology matching  

Limitations:

- Not perfect real-world alignment  
- Some visual-text mismatches possible  
- Serves as a strong **proof-of-concept**, not clinical deployment  

---

# 4. System Architecture

The system is structured into **two modeling phases**:

---

## 4.1 Phase 1 — Supervised Encoder-Based Models

### **Text-Only Model**
- Encoder: DistilBERT (768-dim embedding)  
- Fine-tuned on SkinCAP caption labels  
- Serves as a baseline for RQ1 and RQ2  

### **Image-Only Model**
- Encoder: ViT-B/16  
- Image patches → transformer layers → pooled embedding  
- Captures color, shape, texture of dermatological reactions  

### **Multimodal Fusion Models**
We evaluate 3 strategies:

#### 1. **Early Fusion (Feature-Level)**
- Concatenate text + image embeddings  
- Joint projection → classifier  
- Learns cross-modal interactions  

#### 2. **Late Fusion (Decision-Level)**
- Each encoding produces separate class logits  
- Weighted sum of probabilities  
- Simpler, more robust  

#### 3. **Gated Fusion (Adaptive Fusion)**
- Learnable gate (0–1)  
- Chooses when to trust text vs image  
- Especially useful with vague captions or ambiguous visuals  

---

## 4.2 Phase 2 — Multimodal LLM Models

### **1. LLaMA-3.2-Vision-Instruct**
- Accepts image + caption with a prompt  
- Zero-shot classification  
- Strong general-purpose reasoning  

### **2. SkinGPT**
- Vision-language LLM specialized for dermatology  
- Stronger at fine-grained rash classification  

LLMs answer RQ1–RQ3 and provide qualitative insights.

---

# 5. Key Findings (Summary)

*(You can expand these after running results.)*

- Text-only surprisingly strong due to detailed captions  
- Image-only weaker but useful for ambiguous symptoms  
- Gated fusion typically outperforms early/late fusion  
- Skin types III–VI show slightly lower accuracy → fairness concern  
- Multimodal LLMs outperform supervised models on robustness but can hallucinate  

---

# 7. Repository Structure

├── main.py # Master experiment runner

├── config.json # Model + dataset configuration

├── normalize_api.py # Prototype inference API

├── encoders.py # ViT + DistilBERT wrappers

├── trainer.py # Training loops

├── evaluators.py # Metrics: Accuracy, Recall@K, MRR

├── rankers.py # Candidate scoring mechanisms

├── pairing.py # Text-image pairing logic

├── ontology_loader.py # UMLS / MedDRA handling

├── data_utils.py # Normalization + cleaning helpers

├── datasets.py # Dataset management

├── logger.py # Logging utility

├── smm4h_loader.py # ADR text loader

├── cadec_loader.py # ADR text loader

├── skincap_loader.py # Dermatology multimodal loader

├── .gitignore # Ignore temp/cache/etc

└── README.md

---

# 6. Running the Project

Install dependencies:

```bash
pip install -r requirements.txt





