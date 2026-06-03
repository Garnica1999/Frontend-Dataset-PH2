#!/bin/bash

# 1. Obtener la ruta absoluta de donde está el script
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BASE_DIR"

# 2. Cargar el entorno de Conda de manera segura dentro de scripts de Bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ia_stable

print_status() {
    echo -e "\e[1;32m[+]\e[0m $1"
}
print_warning() {
    echo -e "\e[1;33m[!]\e[0m $1"
}

# =============================================================================
# FUNCIÓN: VERIFICACIÓN E INSTALACIÓN DE DEPENDENCIAS (MODERNA Y SIN WARNINGS)
# =============================================================================
if [ -f "requirements.txt" ]; then
    print_status "Verificando dependencias del archivo requirements.txt..."
    
    # Usamos importlib.metadata para validar la presencia de los paquetes instalados
    # y packaging para parsear correctamente cualquier sintaxis del requirements.txt
    FALTAN_DEPS=$(python3 -c "
import sys
from importlib.metadata import version, PackageNotFoundError

# Forzar a pip/python a no mostrar warnings de librerías obsoletas en el script
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

missing = False
try:
    # Intentamos usar la librería estándar o packaging para leer los requisitos
    from packaging.requirements import Requirement
    
    with open('requirements.txt') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                req = Requirement(line)
                # Verificar si el paquete base está instalado en el entorno
                version(req.name)
            except PackageNotFoundError:
                missing = True
                break
            except Exception:
                # Si falla el parseo complejo de packaging, usamos un fallback simple
                pkg_name = line.split('==')[0].split('>=')[0].strip()
                try:
                    version(pkg_name)
                except PackageNotFoundError:
                    missing = True
                    break
    print('MISSING' if missing else 'OK')
except Exception:
    print('MISSING')
")

    if [ "$FALTAN_DEPS" == "MISSING" ]; then
        print_warning "Se detectaron dependencias faltantes o desactualizadas. Instalando..."
        pip install -r requirements.txt
        if [ $? -eq 0 ]; then
            print_status "¡Dependencias instaladas correctamente!"
        else
            print_warning "Hubo un problema instalando algunas dependencias. Intentando continuar..."
        fi
    else
        print_status "Todas las dependencias están al día. Omitiendo instalación."
    fi
else
    print_warning "No se encontró el archivo requirements.txt en la raíz. Omitiendo verificación."
fi
# =============================================================================

# 3. Crear el nombre del archivo log con la fecha y hora actual
FECHA_HORA=$(date +"%d-%m-%Y_%H-%M-%S")
LOG_DIR="$BASE_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/uvicorn_$FECHA_HORA.log"

print_status "Iniciando servidor FastAPI en segundo plano..."
print_status "Registros redirigidos a: logs/uvicorn_$FECHA_HORA.log"

# 4. Levantar Uvicorn en segundo plano (&) y evitar que muera al cerrar la terminal (nohup)
nohup uvicorn backend.app:app --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &

# Guardar el PID (Process ID) del proceso en un archivo oculto para el script de stop
echo $! > .uvicorn.pid

print_status "¡Servidor levantado con éxito! El control ha sido devuelto a la consola."
