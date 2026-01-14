from search import search_prompt
import sys

def main():
    """CLI chat interface for the RAG system."""
    print("="*50)
    print("        BEM-VINDO AO CHAT DO DESAFIO RAG")
    print("="*50)
    print("Iniciando sistema... Por favor, aguarde.")
    
    # Initialize the RAG chain
    chain = search_prompt()

    if not chain:
        print("\n[ERRO] Não foi possível iniciar o chat.")
        print("Certifique-se de que:")
        print("1. O Docker Compose está rodando (docker-compose up -d)")
        print("2. O arquivo .env está configurado com as chaves de API")
        print("3. Você já rodou a ingestão (python src/ingest.py)")
        return
    
    print("\n[PRONTO] Sistema carregado com sucesso!")
    print("Digite sua pergunta abaixo ou 'sair' para encerrar.")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\nVocê: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['sair', 'exit', 'quit']:
                print("\nEncerrando chat. Até logo!")
                break
                
            if not user_input:
                continue
                
            # Process question and show loading state
            sys.stdout.write("Buscando resposta baseada no contexto...")
            sys.stdout.flush()
            
            response = chain.invoke(user_input)
            
            # Clear loading message and print response
            sys.stdout.write("\r" + " " * 45 + "\r")
            print(f"Assistente: {response}")
            
        except KeyboardInterrupt:
            print("\n\nEncerrando chat. Até logo!")
            break
        except Exception as e:
            print(f"\n[ERRO] Ocorreu um problema ao processar sua pergunta: {e}")

if __name__ == "__main__":
    main()
