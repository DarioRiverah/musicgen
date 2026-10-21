import torch
from scipy.io.wavfile import write
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL_ID = "facebook/musicgen-small"

# Usa la GPU si está disponible, si no cae a CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

PROMPT = "Traditional Colombian vallenato, upbeat paseo rhythm, bright accordion melody, rhythmic guacharaca scratching, steady caja vallenata percussion, acoustic bass, joyful and nostalgic folk dance vibe, 110 BPM"

print(f"Dispositivo: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("Cargando MusicGen...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=DTYPE)
model = model.to(DEVICE)
model.eval()

print("Preparando el prompt...")
inputs = processor(text=[PROMPT], padding=True, return_tensors="pt").to(DEVICE)

print("Generando música...")
with torch.inference_mode():
    audio = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=1500,  # ~30 segundos (50 tokens = 1 s)
    )

sample_rate = model.config.audio_encoder.sampling_rate
waveform = audio[0, 0].cpu().float().numpy()

write("musica.wav", sample_rate, waveform)
print("Música generada: musica.wav")