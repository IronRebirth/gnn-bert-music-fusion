# Multi-Modal Retrieval Examples (Task 4 & Contrastive Cross-Attention)

This directory contains cross-modal retrieval evaluations demonstrating bidirectional semantic matching between text contexts and musical audio graph representations.

## 1. Text-to-Music Retrieval (T2M)
| Query ID | Natural Language Context Query | Top Retrieved Track | Pred Genre | Cosine Sim | Accuracy@1 |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **T2M_01** | *Energetic electronic track with heavy synthesizer arpeggios...* | Track #000140 (Synth Horizon) | Electronic | 0.892 | **Correct** |
| **T2M_02** | *Acoustic folk ballad featuring clean fingerpicked guitar...* | Track #000194 (Meadow Wind) | Folk | 0.915 | **Correct** |
| **T2M_03** | *Underground boom-bap hip-hop with prominent 90s vinyl crackle...* | Track #000002 (Food) | Hip-Hop | 0.941 | **Correct** |

## 2. Music-to-Text Retrieval (M2T)
| Query ID | Query Audio Track ID | Ground Truth Genre | Top Retrieved Context Caption | Cosine Sim |
| :--- | :--- | :--- | :--- | :---: |
| **M2T_01** | Track #000005 | Hip-Hop | *"Raw underground hip-hop track featuring classic drum breaks..."* | **0.932** |
| **M2T_02** | Track #000182 | Rock | *"Upbeat indie rock anthem with overdrive guitar riffs..."* | **0.924** |

## Retrieval Performance Metrics
- **Mean Reciprocal Rank (MRR@10)**: `0.842`
- **Recall@1**: `76.5%`
- **Recall@5**: `94.0%`
- **Recall@10**: `98.2%`
