# 🥁 DrumScribe Studio

Aplicação web completa (Frontend + Backend Python) para baixar músicas do YouTube, separar instrumentos com **Demucs**, transcrever a bateria para partitura **MusicXML**, alinhar a letra cantada acima de cada compasso via **Whisper** e estudar no celular ou PC com player interativo (**AlphaTab** + mixer multifaixas).

## 📁 Estrutura do Repositório

- `app.py` — Backend FastAPI com fila de processamento em segundo plano e endpoints de download.
- `processor.py` — Pipeline de áudio (`yt-dlp` → `Demucs` → `ADTOF-pytorch` → `Whisper` + `music21`). O ADTOF identifica bumbo, caixa, tom, chimbal e ataque sobre a faixa de bateria separada pelo Demucs.
- `index.html` — Interface responsiva com renderizador AlphaTab, linha vermelha de acompanhamento, mixer Mute/Solo, controle de velocidade, modo leitura e botão *Tela Acesa* para celular.
- `score-format.js` — Organiza as partituras geradas pelo app em duas vozes de bateria para leitura: pratos acima, bumbo/caixa/tom abaixo. A formatação visual preserva os instantes e instrumentos dos golpes.
- `drumscribe_colab_backend.ipynb` — Notebook atualizado para instalar o ADTOF e iniciar o servidor no Google Colab.
- `Dockerfile` — Configuração pronta para deploy gratuito com FFmpeg pré-instalado.
- `requirements.txt` — Dependências Python.

## 🎧 Gerar novamente com ADTOF no Colab

Abra `drumscribe_colab_backend.ipynb` no Google Colab e execute as duas células, na ordem. A primeira atualiza o código do GitHub e instala o ADTOF; a segunda inicia o servidor e mostra o link do site conectado. Se estiver usando uma cópia antiga do notebook, adicione à primeira célula, após os outros comandos `pip install`:

```python
!pip install -q "git+https://github.com/xavriley/ADTOF-pytorch.git@85c192e78f716ea0b111cc8a5ee4a8f6a3a4f8a9"
```

Execute `!git pull` na pasta `/content/drumscribe-studio` e reinicie a célula do servidor. Partituras feitas antes da instalação do ADTOF precisam ser geradas novamente para receber as notas do modelo. Mesmo com o novo modelo, a separação da bateria e a transcrição podem cometer erros, especialmente em músicas com instrumentos que vazam para a faixa de bateria.

O BPM é refinado pela média dos intervalos entre batidas detectadas. Isso evita que uma pequena diferença de andamento se acumule e desloque as notas após vários compassos. A música “O Leão E A Igreja” foi estimada inicialmente em 129,2 BPM, mas seus intervalos indicam 128 BPM. A partitura dessa música já salva usa a estimativa antiga; gere-a novamente para aplicar a correção.

O cursor da partitura segue o áudio usando o BPM estimado e o início do primeiro compasso salvo com a nova transcrição. Em arquivos antigos ou MIDI importado, o cursor parte do início do áudio; nesses casos, a posição pode não coincidir com a gravação.

No estúdio, a partitura abre com **2 compassos por linha**. O seletor discreto **Compassos/linha** permite escolher 1, 2 ou 4 e guarda a preferência no navegador. As partituras antigas recebem a divisão em vozes na exibição; as novas também são salvas assim no MusicXML. Os golpes e seus instantes não são alterados pela formatação.

As caixas detectadas com intensidade muito menor que as caixas próximas são marcadas com **parênteses** na partitura como possíveis notas fantasmas. O ADTOF detecta o golpe de caixa; a classificação de intensidade usa o áudio separado pelo Demucs. É uma estimativa: pratos ou outros sons simultâneos podem esconder uma nota fantasma, e golpes fracos detectados incorretamente podem receber a marcação. Partituras já salvas precisam ser geradas novamente para receber essa notação.

O modelo ADTOF foi apresentado por Zehren, Alunno e Bientinesi e distribuído sob licença CC BY-NC-SA 4.0 no projeto original. A integração usa o port PyTorch de `xavriley/ADTOF-pytorch`.

## 🚀 Como publicar grátis usando seu GitHub (sem precisar de PC)

1. Crie um repositório novo no seu **GitHub** (ex.: `drumscribe-studio`) e envie todos os arquivos desta pasta.
2. Acesse **[huggingface.co/spaces](https://huggingface.co/spaces)** e clique em **Create new Space**.
3. Escolha:
   - **SDK:** `Docker` (Blank)
   - **Hardware:** `CPU basic` (gratuito) ou GPU (se desejar maior velocidade)
4. Conecte o seu repositório do GitHub ao Space (ou faça upload dos arquivos direto pela aba *Files* do Space).
5. O build será executado automaticamente. Ao finalizar, você terá uma URL pública para abrir direto no navegador do seu celular.

> **Dica:** Na aba **Abrir do Celular / Google Drive** do site, você também pode carregar diretamente arquivos `.musicxml` e `.wav`/`.mp3` que tenha gerado anteriormente no Google Colab e salvado no seu Google Drive.

## 📱 Estudar offline longe de casa

O app pode funcionar como um PWA para **ler partituras já salvas e tocar os áudios baixados**, sem o PC ligado e sem internet. A geração de músicas novas continua exigindo um servidor de IA e conexão.

1. Com internet no celular, abra a página HTTPS do projeto no GitHub Pages. Se ainda não estiver publicada, em **Settings → Pages** selecione **Deploy from a branch → main → / (root)**; depois abra `https://gabriel20gvn.github.io/drumscribe-studio/`.
2. Aguarde aparecer **“Aplicativo pronto para abrir sem internet”** na biblioteca. Use **Adicionar à tela inicial** no menu do navegador para abrir como app.
3. Na biblioteca, toque em **⬇ Salvar no celular** nas músicas desejadas. Espere a mensagem **“pronto para estudar offline”**; ela confirma que MusicXML e faixas de áudio foram guardados no próprio celular.
4. Ative o modo avião e abra o app pelo ícone. As músicas marcadas **📱 Pronto offline** permanecem disponíveis para leitura e reprodução. Faça esse teste antes de sair de casa.

Os arquivos ficam no armazenamento deste navegador e podem desaparecer se seus dados forem apagados ou se o sistema liberar espaço. A biblioteca na nuvem e o processamento da IA não funcionam sem internet. O AlphaTab 1.9.0 e suas fontes foram incluídos localmente, com suas licenças em `vendor/alphatab/`.
