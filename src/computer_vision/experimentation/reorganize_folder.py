import shutil
import zipfile
from pathlib import Path

def prepare_cvat_zip(source_data_path: str, output_zip_name: str):
    """
    Reorganiza as predições do YOLO e cria um ZIP no formato exato 
    exigido pelo CVAT (Ultralytics YOLO format).
    """
    source_dir = Path(source_data_path)
    
    # 1. Cria uma pasta temporária com a estrutura rigorosa do CVAT
    temp_dir = Path("CVAT_TEMP")
    labels_train_dir = temp_dir / "labels" / "train"
    labels_train_dir.mkdir(parents=True, exist_ok=True)
    
    print("Copiando data.yaml...")
    yaml_path = source_dir / "data.yaml"
    if yaml_path.exists():
        shutil.copy(yaml_path, temp_dir / "data.yaml")
    else:
        print("AVISO: data.yaml não encontrado na raiz!")

    print("Coletando arquivos .txt de todas as pastas GoLive...")
    txt_count = 0
    
    # Procura por todos os arquivos .txt dentro de Train/*/labels/
    # O padrão '**' ajuda a buscar recursivamente
    for txt_file in source_dir.glob("Train/*/labels/*.txt"):
        # Copia o arquivo .txt para a pasta labels/train/ unificada
        # Atenção: Se houver imagens com o mesmo nome em GoLives diferentes (ex: frame1.txt), 
        # um vai sobrescrever o outro. Se isso for o caso, precisaremos renomear as imagens no CVAT também.
        dest_file = labels_train_dir / txt_file.name
        shutil.copy(txt_file, dest_file)
        txt_count += 1

    print(f"{txt_count} arquivos .txt copiados com sucesso.")

    # 2. Zipando a pasta temporária
    print(f"Gerando o arquivo {output_zip_name}...")
    with zipfile.ZipFile(output_zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in temp_dir.rglob("*"):
            if file_path.is_file():
                # arcname garante que a estrutura dentro do zip comece na raiz (sem a pasta CVAT_TEMP)
                arcname = file_path.relative_to(temp_dir)
                zipf.write(file_path, arcname)

    # 3. Limpando a bagunça
    shutil.rmtree(temp_dir)
    print("Processo concluído! O arquivo ZIP está pronto para o CVAT.")

if __name__ == "__main__":
    # Ajuste o caminho da sua pasta 'Data' se necessário
    prepare_cvat_zip(source_data_path="dataset", output_zip_name="upload_para_cvat.zip")