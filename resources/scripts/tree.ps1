$Exclude = @('.venv', '.git', '.streamlit', '__pycache__')

function Show-DirectoryTree {
    param([string]$DirectoryPath, [string]$Indent = "")
    $filesAndFolders = @(Get-ChildItem -LiteralPath $DirectoryPath | Where-Object { $Exclude -notcontains $_.Name })
    
    for ($i = 0; $i -lt $filesAndFolders.Count; $i++) {
        $item = $filesAndFolders[$i]
        $isLast = ($i -eq ($filesAndFolders.Count - 1))
        
        if ($isLast) {
            $connector = "└── "
            $spaces = "    "
        } else {
            $connector = "├── "
            $spaces = "│   "
        }
        
        Write-Host "$Indent$connector$($item.Name)"
        
        if ($item.PSIsContainer) {
            $nextIndent = $Indent + $spaces
            Show-DirectoryTree -DirectoryPath $item.FullName -Indent $nextIndent
        }
    }
}

Show-DirectoryTree -DirectoryPath (Get-Location)