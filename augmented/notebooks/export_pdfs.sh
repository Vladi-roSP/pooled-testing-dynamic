#!/usr/bin/env bash
#
# export_pdfs.sh — exporta notebooks de sesion (ya ejecutados) a PDF via nbconvert.
#
# Uso:
#   ./export_pdfs.sh                       # exporta la lista por defecto
#   ./export_pdfs.sh sesion_francisco      # exporta augmented/notebooks/18_sesion_francisco.ipynb
#   ./export_pdfs.sh nb1 nb2 nb3           # varios (sin extension .ipynb)
#
# Por cada notebook imprime OK/FALLO y el tamano del PDF resultante.
# Es robusto: si uno falla, continua con el resto y termina con codigo != 0.
#
set -u

# --- Rutas fijas del entorno -------------------------------------------------
TINYTEX_BIN="/Users/hectorbecerrilvillamil/Library/TinyTeX/bin/universal-darwin"
PYTHON="/Users/hectorbecerrilvillamil/miniconda3/bin/python3.13"

# nbconvert --to pdf usa xelatex por debajo: hay que exponer el bin de TinyTeX.
export PATH="${TINYTEX_BIN}:${PATH}"

# --- Localizar la raiz del repo y el dir de notebooks ------------------------
# El script vive en augmented/notebooks/ ; la raiz del repo es dos niveles arriba.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NB_REL="augmented/notebooks"
cd "${REPO_ROOT}"

# --- Lista por defecto (notebooks de sesion recientes) -----------------------
DEFAULT_NOTEBOOKS=(sesion_francisco avances_post_sesion)

if [ "$#" -gt 0 ]; then
    NOTEBOOKS=("$@")
else
    NOTEBOOKS=("${DEFAULT_NOTEBOOKS[@]}")
fi

# --- Exportacion -------------------------------------------------------------
fail_count=0
ok_count=0

human_size() {
    # tamano legible del archivo pasado como $1 (o "-" si no existe)
    if [ -f "$1" ]; then
        du -h "$1" | cut -f1
    else
        echo "-"
    fi
}

echo "Repo root : ${REPO_ROOT}"
echo "Notebooks : ${NB_REL}"
echo "Python    : ${PYTHON}"
echo "TinyTeX   : ${TINYTEX_BIN}"
echo "-----------------------------------------------------------------------"

for name in "${NOTEBOOKS[@]}"; do
    ipynb="${NB_REL}/${name}.ipynb"
    pdf="${NB_REL}/${name}.pdf"

    if [ ! -f "${ipynb}" ]; then
        printf "FALLO  %-32s  (no existe %s)\n" "${name}" "${ipynb}"
        fail_count=$((fail_count + 1))
        continue
    fi

    # Ejecutar nbconvert; capturar salida por si hay que diagnosticar.
    if "${PYTHON}" -m jupyter nbconvert --to pdf "${ipynb}" > "/tmp/export_${name}.log" 2>&1; then
        if [ -f "${pdf}" ]; then
            printf "OK     %-32s  %s\n" "${name}" "$(human_size "${pdf}")"
            ok_count=$((ok_count + 1))
        else
            printf "FALLO  %-32s  (nbconvert dijo OK pero no hay PDF)\n" "${name}"
            fail_count=$((fail_count + 1))
        fi
    else
        printf "FALLO  %-32s  (ver /tmp/export_${name}.log)\n" "${name}"
        fail_count=$((fail_count + 1))
    fi
done

echo "-----------------------------------------------------------------------"
echo "Resumen: ${ok_count} OK, ${fail_count} FALLO(s)."

# Codigo de salida != 0 si algo fallo, para que sea usable en pipelines.
[ "${fail_count}" -eq 0 ]
