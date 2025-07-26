import logging
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from app.core.config import settings

logger = logging.getLogger("email_service")

def send_reset_email(email: str, code: str):
    try:
        # Validate API key exists
        if not settings.SMTP_PASSWORD:
            logger.error("SendGrid API key is not configured")
            return

        # Create HTML content first
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background: #0a1f44; color: white; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: linear-gradient(135deg, #0a1f44 0%, #0f2c59 100%); padding: 40px; border-radius: 20px;">
                <h1 style="color: #00d0ff; text-align: center;">DeepDiagnose</h1>
                <h2 style="text-align: center;">Password Reset Request</h2>
                <p>You requested a password reset. Use the code below:</p>
                <div style="background: rgba(0, 208, 255, 0.1); padding: 20px; text-align: center; border-radius: 10px; margin: 20px 0;">
                    <h1 style="font-family: 'Courier New', monospace; font-size: 2em; letter-spacing: 5px; color: #00d0ff; margin: 0;">{code}</h1>
                </div>
                <p style="color: #ff4757;"><strong>This code expires in {settings.RESET_CODE_EXPIRY_MINUTES} minutes.</strong></p>
                <p>If you didn't request this, please ignore this email.</p>
            </div>
        </body>
        </html>
        """
        
        # Create SendGrid mail object
        message = Mail(
            from_email=settings.SMTP_FROM_EMAIL,
            to_emails=email,
            subject="DeepDiagnose - Password Reset Code",
            html_content=html_content
        )
        
        # Initialize SendGrid client
        sg = SendGridAPIClient(api_key=settings.SMTP_PASSWORD)
        
        # Send email
        response = sg.send(message)
        
        # Check response status
        if response.status_code == 202:
            logger.info(f"Password reset email sent to {email}")
        else:
            logger.error(f"SendGrid error: {response.status_code} - {response.body.decode('utf-8')}")
            
    except Exception as e:
        logger.exception(f"Failed to send email to {email}: {str(e)}")