import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import whisperx
import gc
import os
import torch
from whisperx.diarize import DiarizationPipeline
from dotenv import load_dotenv

load_dotenv()

def transcribe_and_diarize_cpu_pipeline(audio_path, output_txt_path, model, align_model, align_metadata, diarize_model):
    print("Шаг 1: Расшифровка аудио в текст...")
    audio = whisperx.load_audio(audio_path)

    # На CPU batch_size=4 обеспечивает баланс между скоростью и стабильностью RAM
    result = model.transcribe(audio, batch_size=4)

    print("Шаг 2: Выравнивание таймстампов под русский язык...")
    result = whisperx.align(result["segments"], align_model, align_metadata, audio, "cpu", return_char_alignments=False)

    print("Шаг 3: Разделение спикеров (Диаризация через Pyannote)...")
    diarize_segments = diarize_model(audio_path, min_speakers=2, max_speakers=4)

    print("Шаг 4: Связывание текста с конкретными спикерами...")
    result = whisperx.assign_word_speakers(diarize_segments, result)

    print(f"Шаг 5: Запись результата в файл {output_txt_path}...")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            start_min = int(segment["start"] // 60)
            start_sec = int(segment["start"] % 60)
            speaker = segment.get("speaker", "UNKNOWN_SPEAKER")
            text = segment["text"].strip()

            line = f"[{start_min:02d}:{start_sec:02d}] {speaker}: {text}\n"
            # print(line, end="")
            f.write(line)

    print("\n[Успешно] Готовый файл сохранен!")


if __name__ == "__main__":
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"
    HF_TOKEN = os.getenv('hf_token')

    CPU_THREADS = 6
    os.environ["OMP_NUM_THREADS"] = str(CPU_THREADS)
    os.environ["MKL_NUM_THREADS"] = str(CPU_THREADS)
    torch.set_num_threads(CPU_THREADS)

    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"  # Квантование, чтобы процессор работал в 2 раза быстрее
    MODEL_VERSION = "turbo"  # tiny, base, small, medium, large-v1, large-v2, large-v3, large, turbo

    # 1. Проверяем наличие папки input
    if not os.path.exists(INPUT_DIR):
        print(f"Папка '{INPUT_DIR}' не найдена. Создаю её. Положите туда аудиофайлы.")
        os.makedirs(INPUT_DIR, exist_ok=True)
        exit()

    audio_files = [f for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))]

    if not audio_files:
        print(f"В папке '{INPUT_DIR}' нет файлов для обработки.")
        exit()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Найдено файлов для обработки: {len(audio_files)}")

    # ==========================================
    # ИНИЦИАЛИЗАЦИЯ МОДЕЛЕЙ ОДИН РАЗ НА СТАРТЕ
    # ==========================================
    print(f"\n=== Инициализация нейросетей на CPU (Потоков: {CPU_THREADS}) ===")

    print("1/3 Загрузка модели WhisperX...")
    loaded_model = whisperx.load_model(
        MODEL_VERSION, DEVICE, compute_type=COMPUTE_TYPE, language="ru", threads=CPU_THREADS
    )

    print("2/3 Загрузка модели выравнивания (Wav2Vec2)...")
    loaded_align_model, align_metadata = whisperx.load_align_model(language_code="ru", device=DEVICE)

    print("3/3 Загрузка Pyannote Diarization...")
    loaded_diarize_model = DiarizationPipeline(token=HF_TOKEN, device=DEVICE)

    print("=== Все модели загружены в ОЗУ и готовы! ===\n")

    # 2. Основной цикл обработки файлов
    for index, audio_name in enumerate(audio_files, start=1):
        AUDIO_FILE = os.path.join(INPUT_DIR, audio_name)
        file_name = os.path.splitext(audio_name)[0]

        # 3. ПРОВЕРКА: есть ли уже готовый файл в папке output?
        already_processed = False
        for existing_file in os.listdir(OUTPUT_DIR):
            if existing_file.startswith(f"{file_name}__") and existing_file.endswith(".txt"):
                already_processed = True
                break

        if already_processed:
            print(f"\n[{index}/{len(audio_files)}] Пропуск: '{audio_name}' уже обработан.")
            continue

        OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{file_name}__{datetime.date.today()}.txt")
        print(f"\n[{index}/{len(audio_files)}] Обработка файла: {audio_name}")

        try:
            start_time = datetime.datetime.now()

            # Запускаем стабильный CPU-конвейер
            transcribe_and_diarize_cpu_pipeline(
                audio_path=AUDIO_FILE,
                output_txt_path=OUTPUT_FILE,
                model=loaded_model,
                align_model=loaded_align_model,
                align_metadata=align_metadata,
                diarize_model=loaded_diarize_model
            )

            execution_time_clean = str(datetime.datetime.now() - start_time).split('.')[0]
            print(f"✅ Время обработки файла {audio_name}: {execution_time_clean}")

        except Exception as e:
            print(f"❌ [Ошибка] Не удалось обработать файл {audio_name}. Причина: {e}")
        finally:
            gc.collect()

    print("\n[Готово] Все файлы успешно обработаны!")