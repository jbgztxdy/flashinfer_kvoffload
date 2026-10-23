# Purpose: docs/mla_host_lru_report.md -> temp/report_lru/pdf/mla_host_lru_report.pdf (without the
#          trailing "数据与复现" section). Usage: bash temp/report_lru/pdf/build.sh
set -e
cd "$(dirname "$0")/../../.."
D=temp/report_lru/pdf
sed '/^## 数据与复现/,$d' docs/mla_host_lru_report.md > $D/report.md
pandoc $D/report.md -f gfm -t html5 -s --metadata pagetitle="远端 KV 稀疏 MLA decode 算子：测试报告" \
  --resource-path=docs --css=$D/report.css --embed-resources -o $D/report.html
google-chrome --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf=$D/mla_host_lru_report.pdf file://$PWD/$D/report.html 2>/dev/null
ls -l $D/mla_host_lru_report.pdf
