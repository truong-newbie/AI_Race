# Medical Ontology AI

Xây dựng hệ thống AI xử lý văn bản y khoa tiếng Việt tự do để phát hiện và chuẩn hóa các khái niệm y tế.

## Mục tiêu

Phát hiện các khái niệm y tế trong văn bản lâm sàng (ghi chú bác sĩ, giấy xuất viện, kết quả xét nghiệm) và ánh xạ với chuẩn ICD-10 và RxNorm.

## Entity Types

- `TRIỆU_CHỨNG` - Triệu chứng bệnh nhân mắc phải
- `TÊN_XÉT_NGHIỆM` - Tên xét nghiệm bệnh nhân thực hiện
- `KẾT_QUẢ_XÉT_NGHIỆM` - Kết quả xét nghiệm (giá trị và đơn vị)
- `CHẨN_ĐOÁN` - Chẩn đoán của bác sĩ
- `THUỐC` - Thuốc điều trị

## Assertions

- `isNegated` - Khái niệm bị phủ định
- `isFamily` - Liên quan đến người nhà/họ hàng
- `isHistorical` - Liên quan đến tiền sử bệnh nhân

## Mapping

- `CHẨN_ĐOÁN` → ICD-10
- `THUỐC` → RxNorm

## Cấu trúc

```
medical-ontology/
├── data/           # Dữ liệu
├── src/            # Source code
├── notebooks/      # Jupyter notebooks
├── configs/        # Configuration files
├── models/         # Model weights
├── outputs/        # Output results
├── scripts/        # Utility scripts
├── tests/          # Unit tests
└── requirements.txt
```

## Cài đặt

```bash
pip install -r requirements.txt
```

## Chạy Baseline

```bash
python -m src.pipeline --input data/test/input --output outputs/
```

## Development

```bash
pytest tests/
```
