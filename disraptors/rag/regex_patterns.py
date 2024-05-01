TICKER = r"\b[A-Z]{2,6}[\.\-\=]*[A-Z0-9]*\b"
VANGUARD_PRODUCTS = r"\bVLB|VGV|VCB|VCN|VCE|VRE|VDY|VIU|VI|VDU|VEF|VA|VE|VIDY|VEE|VXC|VVO|VMO|VVL|VFV|VSP|VGG|VGH|VUS|VBU|VRIF|VGAB|VAB|VSB|VSC|VBG|VCIP|VCNS|VBAL|VGRO|VEQT|VUN|VIC100|VIC200|VIC300|VIC400|VIC500|VIC600\b"
TICKER_REGEX_EDGE_CASES = [
    "ETF",
    "MER",
    "BMO",
    "CIBC",
    "TD",
    "RRSP",
    "TFSA",
    "US",
    "DCA",
]
BULLETS = r"[^a-zA-Z_\s]"
