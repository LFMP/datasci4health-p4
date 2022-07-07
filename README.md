# Projeto 4 – Classificação de lesões de substância branca no Lúpus

## Project 4 – Classification of white matter lesions in Lupus

## Apresentação

O presente projeto foi originado no contexto das atividades da disciplina de pós-graduação [*Ciência e Visualização de Dados em Saúde*](https://ds4h.org), oferecida no primeiro semestre de 2022, na Unicamp.

> |Nome  | RA | Especialização
> |--|--|--|
> | Luiz F. M. Pereira  | 103491  | Computação|

## Introdução

> Apresentação de forma resumida do problema (contexto) e a pergunta que se quer responder.

### Ferramentas

As bibliotecas utilizadas no desenvolvimento desse projeto são listadas abaixo.

* Matplotlib
* Numpy
* Pandas
* Plotly
* Pytorch
* Torchvision
* Umap
* ClassLabel
* Linformer
* SkLearn
* TQDM
* vit_pytorch

### Preparo e uso dos dados

O pré-processamento de dados ocorreu de forma diferente para os diferentes conjuntos. O conjunto disponibilizado no início do projeto foi chamado de Conjunto Base, o Conjunto de Teste corresponde ao conjunto de dados utilizado para avaliar o modelo gerado, e o Conjunto SLE corresponde ao conjunto de dados de lesões de substância branca no cérebro.

As máscaras dos conjuntos foram binarizadas usando o Thresholding de Otsu. Os rótulos foram binarizados usando o método ClassLabel, assim o rótulo 'AVC' foi transformado para 0 e o rótulo 'EM' foi transformado para 1. Para os conjuntos Base e Teste as seguintes operações foram realizadas na imagem base:

  1. Reajuste de contraste automático;

  2. Transformação da imagem em tons de cinza para RGB por meio da cópia de canais;

  3. Redimensionamento para o tamanho de entrada do modelo.

Para o Conjunto SLE foi realizado o mesmo procedimento após a normalização da imagem. A extração de atributos das imagens foi realizada utilizando o [Linformer](https://arxiv.org/abs/2006.04768). O diagrama a seguir sumariza as etapas mencionadas acima.

![imagem](reports/pre-process.drawio.png)

## Metodologia

O Conjunto Base foi dividido em dois conjuntos: um de treino e outro de validação. A proporção utilizada foi 80% dos pacientes para o conjunto de treino e 20% dos pacientes para o conjunto de validação. O Conjunto Teste e o Conjunto SLE foram fornecidos separadamente e usados para teste e transferência de conhecimento.

O modelo escolhido para tarefa de classificação foi um Visual Transformer (ViT) com uma implementação mais eficiente de Encoder, o Linformer, e extração de Embeddings usando Extractor. A imagem abaixo ilustra o pipeline utilizado nesse projeto.

![imagem](reports/metodo.drawio.png)

O Linformer possui alguns parâmetros configuráveis que vale a pena destacar, são eles:

* dim: a dimensão do espaço de projeção;
* depth: a quantidade de camadas que o Encoder possui;
* heads: a quantidade de cabeças (correspondente a um Encoder) que analisará a imagem;

de acordo com o artigo do [Linformer](https://arxiv.org/abs/2006.04768), 10 cabeças são o suficiente para qualquer imagem, neste projeto optamos por 8 cabeças, para dim e depth foram utilizados os valores padrões, 512 e 12, respectivamente. Quanto maior o valor para esses parâmetros, maior é o tempo necessário para treinamento e predição do modelo.

Para o modelo ViT também precisamos escolher alguns parâmetros, dentre eles destacamos `image_size` e `path_size`. Para acelerar o treinamento do modelo, fizemos o downsampling na etapa de pré-processamento, o valor escolhido foi 224x224, valor comum de entrada para os ViT. Como estamos trabalhando com imagens pequenas e em um cenário em que queremos prestar atenção em cada pequeno pedaço, definimos um `path_size` de 8 pixels, assim o Transformer conseguirá se ater a pequenos detalhes em cada pedaço da imagem. A quantidade de épocas definidas para treinamento foi de 50, uma vez que modelos de atenção conseguem extrair características de maneira eficiente até mesmo em cenários few-shot learning. Como métrica de avaliação usamos o loss do modelo, para seu cálculo usamos a implementação do Pytorch de [CrossEntropyLoss](https://pytorch.org/docs/stable/nn.html#crossentropyloss), e a acurácia do modelo. Além disso, coletamos as embeddings da penúltima camada do modelo, fizemos uma redução de dimensionalidade utilizando UMAP e plotamos os dados em espaços de 2 e 3 dimensões.

* resultados do treinamento do classificador usando tabelas e gráficos
>
> Justificar as escolhas.
> Esta parte do relatório pode ser copiada da Atividade 11, caso o grupo opte por usar o SVM já treinado.

## Resultados Obtidos e Discussão

> Esta seção deve apresentar o resultado de predição das lesões de LES usando o classificador treinado. Também deve tentar explicar quais os atributos relevantes usados na classificação obtida
>
> * apresente os resultados de forma quantitativa e qualitativa
> * tenha em mente que quem irá ler o relatório é uma equipe multidisciplinar. Descreva questões técnicas, mas também a intuição por trás delas.
>
## Conclusão

> Destacar as principais conclusões obtidas no desenvolvimento do projeto.
>
> Destacar os principais desafios enfrentados.
>
> Principais lições aprendidas.
>
> Trabalhos Futuros:
>
> * o que poderia ser melhorado se houvesse mais tempo?
>
## Referências Bibliográficas

> Lista de artigos, links e referências bibliográficas (se houver).
>
> Fiquem à vontade para escolher o padrão de referenciamento preferido pelo grupo.
