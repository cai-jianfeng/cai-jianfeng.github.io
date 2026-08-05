$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot
$env:JEKYLL_ENV = "development"
chcp 65001 | Out-Null

$bundleCommand = Get-Command bundle -ErrorAction SilentlyContinue
if ($bundleCommand) {
  $bundlePath = $bundleCommand.Source
} elseif (Test-Path -LiteralPath "C:\Ruby33-x64\bin\bundle.bat") {
  $bundlePath = "C:\Ruby33-x64\bin\bundle.bat"
  $env:Path = "C:\Ruby33-x64\bin;$env:Path"
} else {
  throw "Bundler was not found. Install Ruby+Devkit and run 'bundle install' first."
}

$rubyBin = Split-Path -Parent $bundlePath
$env:BUNDLE_SYSTEM_BINDIR = $rubyBin
$env:Path = "$rubyBin;$env:Path"

& $bundlePath exec jekyll serve `
  --livereload `
  --force_polling `
  --config "_config.yml,_config.dev.yml"
