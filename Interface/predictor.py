import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import sys
import shutil
import tempfile
import threading

import cv2

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

CHECKPOINT_PATH = os.path.join(project_root, 'ActualModels', 'best-epoch=22-val_loss=0.2484.ckpt')

from Models import ResNet18ClassifierBackbone


class ImagePredictor:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ResNet18ClassifierBackbone(num_classes=2).to(self.device)
        self._inference_lock = threading.Lock()

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            self.model.load_state_dict(new_state_dict)
        else:
            self.model.load_state_dict(checkpoint)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Расширения видео, которые пытаемся обрабатывать через OpenCV
        self.video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.mpeg', '.mpg', '.m4v', '.wmv'}

    def _predict_from_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        confidence_class1 = self._predict_ai_probability(image)
        predicted_class = 1 if confidence_class1 >= 0.5 else 0
        return predicted_class, confidence_class1

    def _predict_from_video(self, video_path, preview_out_path):
        pred_class, avg_ai_prob = self._predict_from_video_with_progress(
            video_path=video_path,
            preview_out_path=preview_out_path,
            every_n_frames=5,
            on_frame_processed=None
        )
        return pred_class, avg_ai_prob

    def _predict_from_video_with_progress(
        self,
        video_path,
        preview_out_path,
        every_n_frames=5,
        on_frame_processed=None,
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видео: {video_path}")

        # Оценка числа кадров (может быть 0 для некоторых кодеков)
        raw_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        estimated_total = 0
        if raw_count > 0:
            estimated_total = max(1, raw_count // every_n_frames)

        if on_frame_processed:
            on_frame_processed(0, estimated_total)

        ai_probs_sum = 0.0
        processed_frames = 0
        frame_idx = 0

        first_frame_image = None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % every_n_frames != 0:
                    frame_idx += 1
                    continue

                # OpenCV: BGR -> RGB -> PIL.Image
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)

                if first_frame_image is None:
                    first_frame_image = image.copy()

                ai_probs_sum += self._predict_ai_probability(image)
                processed_frames += 1
                frame_idx += 1

                total_for_callback = estimated_total if estimated_total > 0 else processed_frames
                if on_frame_processed:
                    on_frame_processed(processed_frames, total_for_callback)
        finally:
            cap.release()

        if processed_frames == 0:
            raise RuntimeError("Не удалось прочитать ни одного кадра из видео")

        avg_ai_prob = ai_probs_sum / processed_frames

        # Превью: первый обработанный кадр
        if preview_out_path and first_frame_image is not None:
            os.makedirs(os.path.dirname(preview_out_path), exist_ok=True)
            first_frame_image.save(preview_out_path, format="JPEG")

        predicted_class = 1 if avg_ai_prob >= 0.5 else 0
        return predicted_class, float(avg_ai_prob)

    def _predict_ai_probability(self, image):
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with self._inference_lock:
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence_class1 = probabilities[0, 1].item()
        return confidence_class1

    def predict(self, media_path, preview_out_path=None, video_progress_callback=None):
        ext = os.path.splitext(media_path)[1].lower()
        if ext in self.video_extensions:
            if video_progress_callback:
                return self._predict_from_video_with_progress(
                    video_path=media_path,
                    preview_out_path=preview_out_path,
                    every_n_frames=5,
                    on_frame_processed=video_progress_callback
                )
            return self._predict_from_video(media_path, preview_out_path)
        return self._predict_from_image(media_path)

predictor = ImagePredictor()


if __name__ == "__main__":
    import sys
    from pprint import pprint

    # Путь по умолчанию — ваше видео
    default_video_path = r"C:\Users\Nemesis\Videos\2026-02-22 16-24-39.mp4"

    if len(sys.argv) >= 2:
        media_path = sys.argv[1]
    else:
        media_path = default_video_path

    print(f"[INFO] Пытаемся обработать: {media_path}")
    if not os.path.exists(media_path):
        print(f"[ERROR] Файл не найден: {media_path}")
        sys.exit(1)

    ext = os.path.splitext(media_path)[1].lower()
    print(f"[INFO] Расширение: {ext}")

    # Куда сохранять превью для видео
    preview_dir = os.path.join(os.getcwd(), "preview_test")
    os.makedirs(preview_dir, exist_ok=True)
    preview_path = os.path.join(preview_dir, "preview.jpg")

    def debug_progress(processed, total):
        print(f"[PROGRESS] processed={processed}, total={total}")

    try:
        if ext in predictor.video_extensions:
            print("[INFO] Распознано как ВИДЕО. Запускаем видеопайплайн...")
            pred_class, conf = predictor.predict(
                media_path,
                preview_out_path=preview_path,
                video_progress_callback=debug_progress,
            )
            print(f"[RESULT] video class={pred_class}, confidence={conf:.4f}")
            if os.path.exists(preview_path):
                print(f"[INFO] Превью сохранено в: {preview_path}")
            else:
                print("[WARN] Превью не было создано.")
        else:
            print("[INFO] Распознано как ИЗОБРАЖЕНИЕ. Запускаем обычный пайплайн...")
            pred_class, conf = predictor.predict(media_path)
            print(f"[RESULT] image class={pred_class}, confidence={conf:.4f}")
    except Exception as e:
        print("[ERROR] Ошибка при обработке:", repr(e))