import requests, os
os.makedirs('assets', exist_ok=True)
urls = [
    'https://assets9.lottiefiles.com/packages/lf20_x62chJ.json',
    'https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json',
    'https://assets3.lottiefiles.com/packages/lf20_ysrn2iwp.json',
]
for i, url in enumerate(urls):
    r = requests.get(url, timeout=10)
    if r.status_code == 200:
        with open(f'assets/animation_{i+1}.json', 'w') as f:
            f.write(r.text)
        print(f'✅ animation_{i+1}.json téléchargé !')
    else:
        print(f'❌ {i+1} failed : {r.status_code}')
