from django.shortcuts import render, HttpResponse
from django.http import HttpResponseRedirect, JsonResponse

from django.conf import settings
from rest_framework.response import Response
from rest_framework import status
import jwt

def login_required(fn):
    # Decorator to check if the user has sent a valid JWT with the request
    # def wrapper_fn(req):
    #     key = req.headers.get("Authorization")
        
    #     # Check if Authorization header is present
    #     if not key:
    #         return Response({"error": "No Token Found"}, status=status.HTTP_401_UNAUTHORIZED)
        
    #     # Check if the token is in the correct format (Bearer <token>)
    #     key = key.split(" ")
    #     if len(key) != 2 or key[0] != "Bearer":
    #         return Response({"error": "Malformed Token"}, status=status.HTTP_401_UNAUTHORIZED)
        
    #     token = key[1]
        
    #     try:
    #         # Decode the JWT token
    #         jwt_data = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS256"])
    #         # Pass the decoded JWT data to the decorated function
    #         return fn(req, jwt_data)
        
    #     except jwt.ExpiredSignatureError:
    #         return JsonResponse({"error": "Token has expired"}, status=status.HTTP_401_UNAUTHORIZED)
    #     except jwt.InvalidTokenError:
    #         return JsonResponse({"error": "Invalid token"}, status=status.HTTP_401_UNAUTHORIZED)
    #     except Exception as e:
    #         return JsonResponse({"error": "Token error: " + str(e)}, status=status.HTTP_401_UNAUTHORIZED)

    # return wrapper_fn
    def decode_jwt_token(request):
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]

            try:
                decoded_data = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=["HS512"])
                print(decoded_data)
                return fn(request,decoded_data)
            except jwt.ExpiredSignatureError:
                return JsonResponse({'error': 'Token expired'})
            except jwt.InvalidTokenError:
                return JsonResponse({'error': 'Invalid token'})
        else:
            return JsonResponse({'error': 'Authorization header missing or invalid'})
    return decode_jwt_token