import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

# ===============================
# ASR용 Dataset 정의
# ===============================
class ASRDataset(Dataset):
    def __init__(self, audio_paths, transcriptions, char_map, transform=None):
        # 오디오 파일 경로 리스트
        self.audio_paths = audio_paths
        
        # 각 오디오에 대응하는 정답 문장 리스트 (텍스트 전사 리스트)
        self.transcriptions = transcriptions
        
        # 문자 → 숫자 인덱스 매핑 딕셔너리
        self.char_map = char_map
        
        # 선택적 오디오 변환 함수
        self.transform = transform

    def __len__(self):
        # 데이터 개수 반환
        return len(self.audio_paths)

    def __getitem__(self, idx):
        # =====================================
        # 1. 오디오 불러오기
        # =====================================
        if isinstance(self.audio_paths[idx], str):
            waveform, sample_rate = torchaudio.load(self.audio_paths[idx])
        else:
            waveform = self.audio_paths[idx]
            sample_rate = 16000  # 샘플레이트 기본값

        # =====================================
        # 2. 오디오 변환 (볼륨 키우기, 노이즈 추가 등의 선택 사항)
        # =====================================
        if self.transform:
            waveform = self.transform(waveform)

        # =====================================
        # 3️. Mel Spectrogram 계산
        # =====================================
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=40
        )(waveform)

        # =====================================
        # 4. 로그 스케일 변환 (사람 귀 특성 반영)
        # =====================================
        log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)

        # =====================================
        # 5️. 텍스트 → 숫자 인덱스로 변환
        # =====================================
        text = self.transcriptions[idx]
        
        # char_map에 존재하는 문자만 숫자로 변환
        text_indices = [self.char_map[c] for c in text if c in self.char_map]
        
        # 리스트 → 텐서 변환
        text_indices = torch.tensor(text_indices, dtype=torch.long)

        # =====================================
        # 6️. 반환값
        # log_mel: 음성 특징
        # text_indices: 정답 텍스트 (숫자 형태)
        # log_mel.shape[2]: 스펙트로그램 시간 길이
        # len(text_indices): 텍스트 길이
        # =====================================
        return log_mel, text_indices, log_mel.shape[2], len(text_indices)


# ===============================
# 문자 사전 생성 함수
# ===============================
def create_char_map():
    # 사용할 문자 집합 정의
    chars = [
        ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g',
        'h', 'i', 'j', 'k', 'l', 'm', 'n',
        'o', 'p', 'q', 'r', 's', 't',
        'u', 'v', 'w', 'x', 'y', 'z'
    ]

    # 문자 → 숫자 매핑 (1부터 시작)
    char_map = {char: idx + 1 for idx, char in enumerate(chars)}
    
    # CTC용 blank 토큰은 0
    char_map['<blank>'] = 0

    # 숫자 → 문자 역매핑
    idx_map = {v: k for k, v in char_map.items()}

    return char_map, idx_map


# ===============================
# 배치 생성 함수 (padding 포함)
# ===============================
def collate_fn(batch):
    
    # batch에서 각각 분리
    specs, texts, spec_lengths, text_lengths = zip(*batch)

    # 가장 긴 길이 찾기
    max_spec_len = max(spec_lengths)
    max_text_len = max(text_lengths)

    batch_size = len(specs)

    # =====================================
    # padding을 위한 빈 텐서 생성
    # =====================================
    batch_specs = torch.zeros(batch_size, specs[0].shape[0],
                              specs[0].shape[1], max_spec_len)
    
    batch_texts = torch.zeros(batch_size, max_text_len, dtype=torch.long)

    # =====================================
    # 실제 데이터 채우기
    # =====================================
    for i in range(batch_size):
        spec_len = spec_lengths[i]
        text_len = text_lengths[i]

        # 스펙트로그램 채우기
        batch_specs[i, :, :, :spec_len] = specs[i]

        # 텍스트 채우기
        batch_texts[i, :text_len] = texts[i]

    # 길이 텐서 변환
    spec_lengths = torch.tensor(spec_lengths, dtype=torch.long)
    text_lengths = torch.tensor(text_lengths, dtype=torch.long)

    return batch_specs, batc_

