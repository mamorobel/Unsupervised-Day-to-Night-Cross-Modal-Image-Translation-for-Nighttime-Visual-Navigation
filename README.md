# Enabling 24-hour Agricultural Robotics: Unsupervised Day-to-Night Cross-Modal Image Translation for Nighttime Visual Navigation

![Link to paper will be updated upon acceptance]()

| Cycle | Clip |
|:-----:|:----:|
| ![Cycle](./assets/cycle1_smaller.png) | ![Clip](./assets/cycle1_smaller.png) |

# AgriNight Dataset

| Farm | Daytime | Nighttime |
|:----:|:-------:|:---------:|
| Strawberry Farm 1 | ![Daytime](./assets/farm1_example.png) | ![Nighttime](./assets/farm1_night.png) |
| Strawberry Farm 2 | ![Daytime](./assets/farm2_example.png) | ![Nighttime](./assets/farm2_night.png) |
| Carrot Field | ![Daytime](./assets/carrot_day.png) | ![Nighttime](./assets/carrot_night.png) |

|            | # Day | # Night | # Rows |
|------------|------:|--------:|-------:|
| **Total**        | 428 | 549 | 20 |
| **Strawberry A** | 181 | 185 | 5 |
| **Strawberry B** | 150 | 185 | 9 |
| **Carrot**       | 97  | 179 | 6 |

**Summary of collected daytime and nighttime images and the number of crop rows covered in each farm.**

|       | Traversable | Non-Traversable |     Other |
|-------|------------:|----------------:|----------:|
| **Day**   |   **15.5%** |       **37.8%** |     46.7% |
| **Night** |       12.2% |           33.9% | **53.9%** |

**Class-wise pixel distribution for daytime and nighttime images in the AgriNight dataset.**

| Farm | Daytime | Converted Nighttime |              Segmentation               |
|:----:|:-------:|:-------------------:|:---------------------------------------:|
| Farm 1 | ![Daytime](./assets/farm1.png) | ![Converted Nighttime](./assets/farm1_converted.png) | ![Segmentation](./assets/farm1_seg.png) |
| Farm 2 | ![Daytime](./assets/farm2.png) | ![Converted Nighttime](./assets/farm2_converted.png) | ![Segmentation](./assets/farm2_seg.png) |

# Unsupervised Day2Night Cross Modal Translation


|              Farm A               |              Farm B               |
|:---------------------------------:|:---------------------------------:|
| ![Farm A](./assets/repo_vid2.gif) | ![Farm B](./assets/repo_vid1.gif) |

**Sample videos of segmentation on converted nighttime images from strawberry farms A & B.**

## Masking Method

![Masking Method](./assets/masking_demo.gif)


## Getting Started

### Environmental Setup

[[Placeholder]]

### Training

* Translation Model

`` command placeholder ``

* Segmentation Model

`` command placeholder ``

### Inference

* Translation Model

`` command placeholder ``

* Segmentation Model

`` command placeholder ``

