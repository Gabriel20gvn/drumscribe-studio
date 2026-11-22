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

O BPM é refinado pela média dos intervalos entre batidas detectadas. Isso evita que uma pequena diferença de andamento se acumule e desloque as notas após vários compassos. A música “O Leão E A Igreja” foi estimada inicialmente em 129,2 BPM, mas seus intervalos indicam 128 BPM; as versões atuais usam 128 BPM.

O cursor da partitura segue o áudio usando o BPM estimado e o início do primeiro compasso salvo com a nova transcrição. Em arquivos antigos ou MIDI importado, o cursor parte do início do áudio; nesses casos, a posição pode não coincidir com a gravação.

A linha vermelha acompanha o relógio da faixa que está tocando e é atualizada a cada quadro da tela. Sua posição é interpolada entre as notas desenhadas pelo AlphaTab, com rolagem apenas ao mudar de linha. O mixer inicia as faixas juntas. Isso torna o movimento contínuo, mas uma música com andamento variável ou golpes transcritos incorretamente ainda pode divergir da partitura.

O início agora considera um golpe claro no áudio da bateria mesmo quando o ADTOF não o detecta. Isso evita começar o compasso no golpe seguinte, como ocorreu na geração de “O Leão E A Igreja” de 22/11. A versão salva foi corrigida de 8,61 s para 8,14 s, preservando os 939 golpes detectados e restaurando o bumbo inicial confirmado pela versão anterior e pelo áudio. A partitura corrigida tem 139 compassos. Se ela já estiver salva offline, abra-a uma vez com internet para atualizar só a partitura; os áudios locais permanecem no aparelho. A detecção automática de golpes e do primeiro tempo ainda pode falhar em músicas com introdução ambígua.

No estúdio, **Compassos/linha: Auto** mostra 2 compassos por linha com o celular em pé e 4 quando ele é girado para a horizontal. Também é possível fixar 1, 2 ou 4 compassos; a escolha fica guardada no navegador. Ao girar o celular, a partitura é redesenhada na nova largura. Na horizontal, o app libera a tela para a leitura, compacta os controles e mostra duas grades por linha. As partituras antigas recebem a divisão em vozes na exibição; as novas também são salvas assim no MusicXML. Os golpes e seus instantes não são alterados pela formatação.

O seletor **Visualização** alterna entre **Partitura** e **Grade**. A grade mostra bumbo, caixa, chimbal, pratos e tons em linhas com a contagem `1 e + a`, traz a letra sobre cada compasso e acompanha o áudio com uma linha rosa. Usa os mesmos golpes do MusicXML, inclusive as possíveis notas fantasmas da caixa, sem gerar uma análise nova. A preferência fica salva no navegador e funciona offline para músicas já salvas no celular. A grade exige MusicXML de bateria em 4/4; MIDI/GP importado continua disponível na partitura tradicional.

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
