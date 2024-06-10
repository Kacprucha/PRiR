# PRiR - Sentiment Analysis with TF-IDF Algorithm

## Project Overview
This repository contains the semester project for the course "Programowanie Równoległe i Rozproszone 2024L" (Parallel and Distributed Programming). The project focuses on sentiment analysis using a custom variation of the TF-IDF algorithm.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Installation
To get started with the project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/Kacprucha/PRiR.git
cd PRiR
pip install -r requirements.txt
```

## Usage
Run the scripts with the file you want to analyze as an argument.

```bash
python sekwencyjny.py <file_to_analyze>
python podzial_domenowy.py <file_to_analyze>
python podzial_funkcjonalny.py <file_to_analyze>
python podzial_funkcjonalny_v2.py <file_to_analyze>
```

## Project Structure
- `biblioteka.py`: Implements TF-IDF using a Python library.
- `podzial_domenowy.py`: Script for domain decomposition.
- `podzial_funkcjonalny.py`: Script for functional decomposition.
- `podzial_funkcjonalny_v2.py`: Program with thread 0 working as a server, handling functional decomposition.
- `sekwencyjny.py`: Sequential version of the sentiment analysis.
- `data/`: Directory containing datasets.
- `teoria.md`: Theoretical background and explanation of the project.
