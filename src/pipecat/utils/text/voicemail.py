
VOICEMAIL_PHRASES = [
    'voice mail', 'voicemail', 'message', 'leave voicemail', 'grabar un mensaje', 'mensaje', 'marcando', 'mensaje predeterminado',
    'contestador', 'deje un mensaje', 'dejar un mensaje', 'mensaje de voz', 'despues del tono', 'oir el tono', 'movistar', 'predeterminado',
    'grabe su mensaje', 'buzon', 'tono', 'finalizar', 'grabe', 'buzon de voz', 'no se encuentra', 'no esta disponible', 'no contesta', 'correo',
    'servicio', 'este es', 'tendra costo', 'de este momento'
]

def is_text_voicemail(text: str):

    content = text.lower().replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace(" ", "")

    for phrase in VOICEMAIL_PHRASES:
        phrase = phrase.replace(" ", "")
        if phrase in content:
            return True
        
    return False





def _test_voicemail():

    text = 'Tendrá costo a partir de este momento'
    is_voicemail = is_text_voicemail(text)
    print(is_voicemail)

if __name__ == '__main__':
    _test_voicemail()