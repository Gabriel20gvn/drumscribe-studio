# 🥁 DrumScribe Studio

Aplicação web completa (Frontend + Backend Python) para baixar músicas do YouTube, separar instrumentos com **Demucs**, transcrever a bateria para partitura **MusicXML**, alinhar a letra cantada acima de cada compasso via **Whisper** e estudar no celular ou PC com player interativo (**AlphaTab** + mixer multifaixas).

## 📁 Estrutura do Repositório

- `app.py` — Backend FastAPI com fila de processamento em segundo plano e endpoints de download.
- `processor.py` — Pipeline de áudio (`yt-dlp` → `Demucs` → transcrição de bateria + `Whisper` + `music21`).
- `static/index.html` — Interface responsiva com renderizador de partitura AlphaTab, mixer Mute/Solo, controle de velocidade, modo leitura e botão *Tela Acesa* para celular.
- `Dockerfile` — Configuração pronta para deploy gratuito com FFmpeg pré-instalado.
- `requirements.txt` — Dependências Python.

## 🚀 Como publicar grátis usando seu GitHub (sem precisar de PC)

1. Crie um repositório novo no seu **GitHub** (ex.: `drumscribe-studio`) e envie todos os arquivos desta pasta.
2. Acesse **[huggingface.co/spaces](https://huggingface.co/spaces)** e clique em **Create new Space**.
3. Escolha:
   - **SDK:** `Docker` (Blank)
   - **Hardware:** `CPU basic` (gratuito) ou GPU (se desejar maior velocidade)
4. Conecte o seu repositório do GitHub ao Space (ou faça upload dos arquivos direto pela aba *Files* do Space).
5. O build será executado automaticamente. Ao finalizar, você terá uma URL pública para abrir direto no navegador do seu celular.

> **Dica:** Na aba **Abrir do Celular / Google Drive** do site, você também pode carregar diretamente arquivos `.musicxml` e `.wav`/`.mp3` que tenha gerado anteriormente no Google Colab e salvado no seu Google Drive.
