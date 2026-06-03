#!/bin/bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$BASE_DIR"

print_status() {
    echo -e "\e[1;34m[*]\e[0m $1"
}
print_error() {
    echo -e "\e[1;31m[-]\e[0m $1"
}

PID_FILE=".uvicorn.pid"

# 1. Verificar si el archivo con el PID existe
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    # 2. Verificar si el proceso realmente sigue vivo en el sistema
    if ps -p $PID > /dev/null; then
        print_status "Deteniendo el servidor Uvicorn de manera segura (PID: $PID)..."
        
        # Enviar señal de terminación segura (SIGTERM)
        kill $PID
        
        # Esperar un momento a que TensorFlow limpie la memoria y cierre la sesión
        sleep 2
        
        # Eliminar el archivo de PID residual
        rm "$PID_FILE"
        print_status "El servidor se ha detenido correctamente."
    else
        print_error "El proceso con PID $PID ya no se encuentra activo."
        rm "$PID_FILE"
    fi
else
    # 3. Plan de respaldo si no encuentra el archivo .pid (busca el proceso por comando)
    print_status "No se encontró el archivo de control pid. Buscando procesos activos de Uvicorn..."
    PID_BACKUP=$(pgrep -f "uvicorn backend.app:app")
    
    if [ ! -z "$PID_BACKUP" ]; then
        print_status "Proceso encontrado mediante búsqueda alternativa (PID: $PID_BACKUP). Deteniendo..."
        kill $PID_BACKUP
        sleep 2
        print_status "Servidor detenido."
    else
        print_error "No se detectó ningún servidor Uvicorn ejecutándose para esta aplicación."
    fi
fi
