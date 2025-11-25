
VOICEMAIL_PHRASES = [
    'este es', 'oir el tono', 'no esta disponible', 'numero', 'no contesta', 
    'buzon', 'movistar', 'message', 'mensaje de voz', 'no se encuentra', 
    'tono', 'voicemail', 'predeterminado', 'grabe su mensaje', 'mensaje', 
    'dejar un mensaje', 'voice mail', 'contestador', 'transferida', 'tendra costo', 
    'de este momento', 'leave voicemail', 'correo', 'finalizar', 
    'mensaje predeterminado', 'deje un mensaje', 'grabe', 
    'buzon de voz', 'marcando', 'grabar un mensaje', 'despues del tono'
]

def is_text_voicemail(text: str):
    print('is_text_voicemail  Verifying if transcripot has voicemail phrases')
    content = text.lower().replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace(" ", "")
    print('is_text_voicemail normalized transcript:', content)
    for phrase in VOICEMAIL_PHRASES:
        phrase = phrase.replace(" ", "")
        if phrase in content:
            print('is_text_voicemail Phrase found:', phrase)
            return True
    print('is_text_voicemail No Phrases found')
    return False





def _test_voicemail():

    text = 'El número móvil marcado esta temporalmente fuera de servicio.'
    is_voicemail = is_text_voicemail(text)
    print(is_voicemail)

if __name__ == '__main__':
    _test_voicemail()