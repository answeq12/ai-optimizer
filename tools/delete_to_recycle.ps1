# 99_불필요파일_삭제.bat 이 실행한다. tools/delete_list.txt 의 항목을 보여 주고, y 를 입력하면 휴지통으로 보낸다.
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName Microsoft.VisualBasic
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$allowed = @($root, 'D:\halio_data')
$protect = @('results\preds', 'results\coco_ret', 'splits', 'jobs\done', 'installers', 'src', 'dfc', 'docs')
Set-Location $root

if (Test-Path (Join-Path $root 'jobs\running.txt')) {
  Write-Host '!! 작업 대기소에서 실험이 돌고 있습니다. 끝난 뒤 다시 실행하세요.' -ForegroundColor Red
  exit 1
}

function SizeOf($item) {
  if ($item.PSIsContainer) {
    $s = (Get-ChildItem -LiteralPath $item.FullName -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    if ($null -eq $s) { return 0 } else { return $s }
  }
  return $item.Length
}

$items = @()
foreach ($line in (Get-Content -Encoding UTF8 (Join-Path $PSScriptRoot 'delete_list.txt'))) {
  $l = $line.Trim()
  if ($l -eq '' -or $l.StartsWith('#')) { continue }
  $p = if ([System.IO.Path]::IsPathRooted($l)) { $l } else { Join-Path $root ($l -replace '/', '\') }
  foreach ($f in (@(Get-Item -Path $p -Force -ErrorAction SilentlyContinue) | Where-Object { $null -ne $_ })) {
    $full = $f.FullName
    $ok = $false
    foreach ($a in $allowed) { if ($full.StartsWith($a + '\', 'OrdinalIgnoreCase')) { $ok = $true } }
    foreach ($q in $protect) { if ($full.TrimEnd('\') -ieq (Join-Path $root $q) -or $full.StartsWith((Join-Path $root $q) + '\', 'OrdinalIgnoreCase')) {
        if (-not $full.Contains('__pycache__')) { $ok = $false } } }
    if ($ok) { $items += $f } else { Write-Host "  (보호 경로라 건너뜀) $full" -ForegroundColor Yellow }
  }
}

if ($items.Count -eq 0) { Write-Host '지울 것이 없습니다.'; exit 0 }

Write-Host ''
Write-Host '휴지통으로 보낼 항목:' -ForegroundColor Cyan
$total = 0
foreach ($f in $items) {
  $s = SizeOf $f; $total += $s
  $kind = if ($f.PSIsContainer) { '[폴더]' } else { '[파일]' }
  Write-Host ('  {0} {1}  ({2:N1} MB)' -f $kind, $f.FullName.Replace($root + '\', ''), ($s / 1MB))
}
Write-Host ('합계 {0} 개, {1:N1} MB' -f $items.Count, ($total / 1MB))
Write-Host ''
$ans = Read-Host '휴지통으로 보낼까요? (y 입력 후 Enter, 그 밖에는 취소)'
if ($ans -ne 'y') { Write-Host '취소했습니다. 아무것도 지우지 않았습니다.'; exit 0 }

$n = 0
foreach ($f in $items) {
  try {
    if ($f.PSIsContainer) {
      [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteDirectory($f.FullName, 'OnlyErrorDialogs', 'SendToRecycleBin')
    } else {
      [Microsoft.VisualBasic.FileIO.FileSystem]::DeleteFile($f.FullName, 'OnlyErrorDialogs', 'SendToRecycleBin')
    }
    $n++
  } catch { Write-Host "  실패: $($f.FullName) - $($_.Exception.Message)" -ForegroundColor Red }
}
Write-Host "끝. $n 개를 휴지통으로 보냈습니다."
