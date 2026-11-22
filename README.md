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

## 📱 Estudar offline longe de casa

O app pode funcionar como um PWA para **ler partituras já salvas e tocar os áudios baixados**, sem o PC ligado e sem internet. A geração de músicas novas continua exigindo um servidor de IA e conexão.

1. Com internet no celular, abra a página HTTPS do projeto no GitHub Pages. Se ainda não estiver publicada, em **Settings → Pages** selecione **Deploy from a branch → main → / (root)**; depois abra `https://gabriel20gvn.github.io/drumscribe-studio/`.
2. Aguarde aparecer **“Aplicativo pronto para abrir sem internet”** na biblioteca. Use **Adicionar à tela inicial** no menu do navegador para abrir como app.
3. Na biblioteca, toque em **⬇ Salvar no celular** nas músicas desejadas. Espere a mensagem **“pronto para estudar offline”**; ela confirma que MusicXML e faixas de áudio foram guardados no próprio celular.
4. Ative o modo avião e abra o app pelo ícone. As músicas marcadas **📱 Pronto offline** permanecem disponíveis para leitura e reprodução. Faça esse teste antes de sair de casa.

Os arquivos ficam no armazenamento deste navegador e podem desaparecer se seus dados forem apagados ou se o sistema liberar espaço. A biblioteca na nuvem e o processamento da IA não funcionam sem internet. O AlphaTab 1.9.0 e suas fontes foram incluídos localmente, com suas licenças em `vendor/alphatab/`.
