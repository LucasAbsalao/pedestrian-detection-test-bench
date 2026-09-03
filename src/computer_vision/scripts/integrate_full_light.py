import pandas as pd
from computer_vision.core.config import EVALUATIONS_DIR, UTILS_DIR

systems = ['blaxtair', 'brigade', 'efa', 'stonkam_camera', 'stonkam_screen']

for system in systems:

    path_geral = EVALUATIONS_DIR / f"{system}_full_test" / f"{system}_full_test.csv"
    path_geral_delay = EVALUATIONS_DIR / f"{system}_full_test" / f"{system}_full_test_delay.csv"

    path_light = EVALUATIONS_DIR / f"{system}_light_test" / f"{system}_light_test.csv"
    path_light_delay = EVALUATIONS_DIR / f"{system}_light_test" / f"{system}_light_test_delay.csv"
    

    df_geral = pd.read_csv(path_geral)
    df_geral_delay = pd.read_csv(path_geral_delay)
    df_light = pd.read_csv(path_light)
    df_light_delay = pd.read_csv(path_light_delay)

    coluna_nome = 'Name' # Troque para 'name' minúsculo se for o caso do seu CSV

    df_light[coluna_nome] = df_light[coluna_nome].apply(
        lambda x: str(x).replace('.mp4', '') + '_light.mp4' if str(x).endswith('.mp4') else str(x) + '_light'
    )

    coluna_nome = 'name'
    df_light_delay[coluna_nome] = df_light_delay[coluna_nome].apply(
        lambda x: str(x).replace('.mp4', '') + '_light.mp4' if str(x).endswith('.mp4') else str(x) + '_light'
    )

    # 2. Concatenar a tabela de luz ofuscante na tabela geral
    # ignore_index=True garante que a numeração das linhas (índice) seja recalculada corretamente
    df_complete = pd.concat([df_geral, df_light], ignore_index=True)
    df_complete_delay = pd.concat([df_geral_delay, df_light_delay], ignore_index=True)

    df_complete.to_csv(str(UTILS_DIR / 'graphs' / f'{system}_complete_test.csv'), index=False)
    df_complete_delay.to_csv(str(UTILS_DIR / 'graphs' / f'{system}_complete_test_delay.csv'), index=False)