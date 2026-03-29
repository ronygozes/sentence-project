test_cases = {
    # Nominal pipe sizing — 1/2" = DN15, NOT 12.7mm (actual OD)
    'צינור PVC לחץ 1/2 אינץ': [
        'צינור PVC PN16 15dn קוטר',       # ✓ match
        'צינור PVC PN16 קוטר 12.7 מ"מ',     # ✗ literal conversion, wrong
        'צינור PVC PN16 קוטר 20 מ"מ',       # ✗ wrong size
    ],
    ' 2צינור PVC לחץ 1/2 אינץ': [
        ' צינור PVC DN15',
        'צינור PVC DN12'
    ],
    # 3/4" = DN20, and material must match
    'צינור גלוון 3/4 אינץ': [
        'צינור גלוון DN20 קוטר נומינלי 20 מ"מ',  # ✓ match
        'צינור גלוון DN25 קוטר נומינלי 25 מ"מ',  # ✗ wrong size
        'צינור נחושת 3/4 אינץ',                 # ✗ wrong material
    ],
    # Thread pitch matters — M8x1.25 ≠ M8x1.0
    'בורג M8x1.25 נירוסטה': [
        'בורג מטרי 8 מ"מ פיצינג 1.25 נירוסטה A2',  # ✓ match
        'בורג M8x1.0 נירוסטה',                         # ✗ different pitch
        'בורג 5/16 אינץ UNC פלדה',                   # ✗ wrong standard + material
    ],
    'כבל חשמל 10 AWG': [
        'כבל 5.26 מ"מ² NYY',   # ✓ match (standard equivalent) 
        'כבל 6 מ"מ² NYY',      # ✗ too small
        'כבל 2.5 מ"מ² NYY',    # ✗ way too small
    ],
    # 100 PSI ≈ 6.9 bar ≈ PN7
    'שסתום כדורי 100 PSI': [
        'שסתום כדורי PN7 (6.9 בר)',   # ✓ match
        'שסתום כדורי PN16 (16 בר)',   # ✗ wrong pressure rating
        'שסתום זווית 100 PSI',         # ✗ wrong valve type
    ],
    # 16 gauge sheet metal ≈ 1.5mm (NOT 1.27mm exact)
    'פח גלוון 16 gauge': [
        'פח גלוון עובי 1.5 מ"מ',    # ✓ match (standard equivalent)
        'פח גלוון עובי 2.0 מ"מ',    # ✗ wrong thickness
        'פח אלומיניום 1.5 מ"מ',     # ✗ wrong material
    ],
    # No match at all — tests null response
    'מרפק PVC 90° 2 אינץ': [
        'מרפק PVC 90° קוטר 50 מ"מ',   # ✓ match
        'מרפק PPR 90° קוטר 50 מ"מ',   # ✗ wrong material
        'מרפק PVC 90° קוטר 72 מ"מ',   # ✗ wrong size
    ],
    'מרפק PVC 90° 2 אינץ שוב': [
        'מרפק PVC 90° קוטר 63 מ"מ',    # ✓ match
        'מרפק PPR 90° קוטר 63 מ"מ',    # ✗ wrong material
        'מרפק PVC 90° קוטר 90 מ"מ',    # ✗ wrong size (DN80, not DN50)
    ],
    'מרפק PVC 90° 2 אינץ נוסף שוב': [
        'מרפק PVC 90° קוטר 63 מ"מ',    # ✓ match
        'מרפק PPR 90° קוטר 63 מ"מ',    # ✗ wrong material
        'מרפק PVC 90° קוטר 50 מ"מ',    # ✗ wrong size (DN80, not DN50)
    ],
    'מרפק PVC 90° 2 אינץ נוסף': [
        'מרפק PVC 45° קוטר 63 מ"מ',    # ✗ wrong angle
        'מרפק PPR 90° קוטר 63 מ"מ',    # ✗ wrong material
        'מרפק PVC 90° קוטר 300 מ"מ',    # ✗ wrong size (DN80, not DN50)
    ],
    'כבל חשמל 10 AWG again': [
        'כבל 6 מ"מ² NYY',    # ✗ too big
        'כבל 4 מ"מ² NYY',    # ✗ too small
        'כבל 2.5 מ"מ² NYY',  # ✗ way too small
    ],
}
