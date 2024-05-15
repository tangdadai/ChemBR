smi_one_char = (
  '#', '(', ')', '-', '/', 'c', 'n', 'o', 's',
  '1', '2', '3', '4', '5', '6', '7', '8', '9', '\\',
  '=', 'C', 'F', 'I', 'N', 'O', 'P', 'S', 'B')
smi_two_char = (
  'Br', 'Cl')
smi_three_char = (
  '%10', '%11')
smi_one_bracket = (
  '[C]', '[N]', '[O]')
smi_two_bracket = (
  '[C@]', '[N+]', '[N-]', '[O+]', '[O-]', '[P+]', '[P@]', '[PH]',
  '[S+]', '[S-]', '[S@]', '[SH]', '[n+]', '[n-]', '[nH]', '[o+]', '[s+]',
  '[B-]', '[C+]', '[C-]', '[CH]', '[I+]', '[N@]', '[Si]', '[Sn]', '[c-]')
smi_three_bracket = (
  '[C@@]', '[C@H]', '[CH-]', '[NH+]', '[NH-]', '[OH+]',
  '[P@@]', '[PH+]', '[PH2]', '[S@@]', '[SH+]', '[nH+]',
  '[B@-]', '[BH-]', '[CH2]', '[IH2]', '[N@+]', '[N@@]', '[P@+]', '[S@+]', '[Si-]', '[cH-]')
smi_four_bracket = (
  '[C@@H]', '[CH2-]', '[NH2+]',
  '[NH3+]', '[P@@H]', '[PH2+]', '[S@@+]',
  '[B@@-]', '[BH2-]', '[BH3-]', '[N@@+]', '[P@@+]', '[Sn+2]', '[Sn+3]')

smi_vocab = ('<Start>', '<Pad>', '<Mask>') + smi_one_char + smi_two_char + smi_three_char\
            + smi_one_bracket + smi_two_bracket + smi_three_bracket + smi_four_bracket

smi_vocab_size = len(smi_vocab)
