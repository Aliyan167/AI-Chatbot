from django.shortcuts import render

# Create your views here.

from rest_framework.decorators import api_view
from rest_framework.response import Response
from chat.hrbp_agent import HRBPAgent

agent = HRBPAgent()


@api_view(['POST'])
def chat_api(request):
    """
    POST request with JSON:
    {
        "message": "Your HR question"
    }
    Returns:
    {
        "reply": "AI response"
    }
    """
    data = request.data
    message = data.get("message")

    if not message:
        return Response({"error": "Message is required"}, status=400)

    try:
        reply = agent.ask(message)
        return Response({"reply": reply})
    except Exception as e:
        return Response({"error": str(e)}, status=500)


from django.shortcuts import render


def chat_page(request):
    return render(request, "chat/index.html")

