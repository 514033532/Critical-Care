#!/bin/bash
# Run the full figure pipeline sequentially (each script loads the annotated object).
set -e
cd /home/zhh2/Jiangjianrong/new-anlysis
mkdir -p logs

for s in fig1_overview fig2_neutrophil fig3_nk fig4_arg1_nk fig5_cd38 fig6_network fig7_metabolism fig8_ml; do
    echo "=== $s ==="
    python3 scripts/${s}.py > logs/${s}.log 2>&1 || { echo "FAILED: $s"; tail -40 logs/${s}.log; exit 1; }
    echo "OK: $s"
done
echo "ALL DONE"
