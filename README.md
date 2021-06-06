# RoI DAER

## Como usar?

- Inicialmente, dois arquivos são os mais essenciais: roi.py e full_analysis.ipynb

- Requerimentos: ``pip install requirements.txt``

** roi.py: **
- Responsável por gerar os arquivos de dados no formato .pkl (pickled) para armazenar os DataFrames
- Variáveis para ajustar:
    - filename: caminho do arquivo de saída do script Yolo + DeepSort (e.g. "RUBEM_BERTA_3_60_6_mars-small128.csv") - a partir da subfolder /data/
    - videoname: caminho do arquivo de video inicial (raw) usado - extrair primeiro frame em 1s p/ compor os dados
- Executar via terminal:
    ``python3 roi.py``

** full_analysis.ipynb: **
- Responsável por puxar os arquivos .pkl 

- Variáveis para ajustar:
    - filename: caminho do arquivo de saída do script Yolo + DeepSort
  
- Executar via Jupyter Notebook / pode ser aberto no colab se tiver os dados no path.