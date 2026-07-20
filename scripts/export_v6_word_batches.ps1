param(
    [Parameter(Mandatory = $true)]
    [string]$InputDocx,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir,

    [ValidateRange(1, 50)]
    [int]$BatchSize = 5,

    [ValidateRange(0, 10000)]
    [int]$PageCount = 0,

    [switch]$SkipFieldRefresh,

    [switch]$DirectFullDocument,

    [switch]$SaveBeforeExport
)

$ErrorActionPreference = "Stop"

$inputPath = (Resolve-Path -LiteralPath $InputDocx).Path
$outputPath = [System.IO.Path]::GetFullPath($OutputDir)
[System.IO.Directory]::CreateDirectory($outputPath) | Out-Null

$word = $null
$document = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.ScreenUpdating = $false

    $document = $word.Documents.Open($inputPath, $false, $false)

    if (-not $SkipFieldRefresh) {
        foreach ($toc in $document.TablesOfContents) {
            $toc.Update()
        }
        foreach ($tableOfFigures in $document.TablesOfFigures) {
            $tableOfFigures.Update()
        }
        $document.Fields.Update() | Out-Null

        foreach ($story in $document.StoryRanges) {
            $range = $story
            while ($null -ne $range) {
                $range.Fields.Update() | Out-Null
                $range = $range.NextStoryRange
            }
        }

        $document.Repaginate()
        $document.Save()
    }

    if ($SaveBeforeExport -and $SkipFieldRefresh) {
        $document.Saved = $false
        $document.Save()
    }

    if ($DirectFullDocument) {
        $filename = "full_document.pdf"
        $pdfPath = Join-Path $outputPath $filename
        $document.ExportAsFixedFormat(
            $pdfPath,
            17,
            $false,
            0,
            0,
            1,
            1,
            0,
            $true,
            $true,
            1,
            $true,
            $true,
            $false
        )
        Write-Output "exported=$filename range=all"
    }
    else {
        $totalPages = $PageCount
        if ($totalPages -eq 0) {
            $document.Repaginate()
            $totalPages = $document.ComputeStatistics(2)
        }

        for ($first = 1; $first -le $totalPages; $first += $BatchSize) {
            $last = [Math]::Min($first + $BatchSize - 1, $totalPages)
            $filename = "V6_pages_{0:D3}_{1:D3}.pdf" -f $first, $last
            $pdfPath = Join-Path $outputPath $filename
            $document.ExportAsFixedFormat(
                $pdfPath,
                17,
                $false,
                0,
                3,
                $first,
                $last,
                0,
                $true,
                $true,
                1,
                $true,
                $true,
                $false
            )
            Write-Output "exported=$filename pages=$first-$last"
        }

        Write-Output "page_count=$totalPages"
    }
}
finally {
    if ($null -ne $document) {
        $document.Close($false)
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::FinalReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
