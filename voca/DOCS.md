# Voca Speech

Local text-to-speech and speech-to-text for German and English, running on the CPU.

- **TTS:** tiny-voca (Fraunhofer-IEM/tiny-voca-de-en), 32 German and 32 English voices
- **TTS:** apptor-tiny-tts (apptor/apptor-tiny-tts), German only, voices `apptor-de-f2` (female) and
  `apptor-de-m2` (male); clearer German pronunciation, spells out times, dates and amounts
- **STT:** Phonon-2.5 (Fermion Research), detects the spoken language itself

## First start

The first start downloads about 0.45 GB of models into the add-on's data folder (updating from 1.0.x
only fetches the ~50 MB apptor model). That needs internet access and takes a minute or two; later
starts work offline. Watch progress in the **Log** tab.

## Use it in Assist

1. After the add-on starts, Home Assistant shows a discovered **Wyoming Protocol** device under
   *Settings → Devices & services*. Click **Configure** to add it.
   (Adding it by hand: Wyoming Protocol integration, host `local-voca`, port `10200`.)
2. *Settings → Voice assistants*: edit your assistant and pick **tiny-voca** as text-to-speech and
   **phonon** as speech-to-text. Choose a voice there, or leave it on the default from the options below.
   For German, the voices `apptor-de-f2` / `apptor-de-m2` (apptor-tiny-tts) are in the same voice list.
   After an update that adds voices, reload the Wyoming entry (⋮ → **Reload**) so the list refreshes.

## Options

| Option | Meaning |
|---|---|
| `voice_de` | Default German voice when Assist doesn't name one (`fresh-de-f00`…`f15`, `fresh-de-m00`…`m15`, `apptor-de-f2`, `apptor-de-m2`) |
| `voice_en` | Default English voice (`fresh-en-…`) |
| `stt_model` | `FermionResearch/Phonon-2.5` (multilingual) or `FermionResearch/Phonon-2.5-English` |
| `apptor_vocos` | Vocos sound filter for the apptor voices (clearer, a bit slower; off sounds buzzy) |
| `log_level` | `debug` logs every recognised sentence and spoken text |

## OpenAI-compatible API

Port 8000 serves `POST /v1/audio/speech` and `POST /v1/audio/transcriptions` (plus `/v1/models`,
`/v1/audio/voices`, `/health`), so other tools on your network can use the same models via
`http://<home-assistant-ip>:8000/v1`. apptor: `"voice": "apptor-de-m2"`, or `"model": "apptor-tiny-tts"`
with an OpenAI voice name (nova, shimmer, … → F2; echo, onyx, … → M2). Remove the port mapping in the add-on's *Network* section
if you don't want that. Not available in the add-on image: `aac` output and m4a/webm input (they need ffmpeg).

## Limits

- German pronunciation goes through espeak-ng mapped onto the model's phonemes (the model's own German
  frontend is unpublished). Long compound words are sometimes slightly off.
- Voice quality varies; some low voices sound metallic.
