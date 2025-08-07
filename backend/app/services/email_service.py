import logging
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from app.core.config import settings

logger = logging.getLogger("email_service")

def send_otp_email(email: str, otp: str):
    """
    Send OTP verification email for signup using SendGrid
    """
    try:
        # Validate API key exists
        if not settings.SMTP_PASSWORD:
            logger.error("SendGrid API key is not configured")
            return

        # Create HTML content with DeepDiagnose branding
        html_content = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background: #0a1f44; color: white; padding: 20px; margin: 0;">
            <div style="max-width: 600px; margin: 0 auto; background: linear-gradient(135deg, #0a1f44 0%, #0f2c59 100%); border-radius: 20px; overflow: hidden; box-shadow: 0 10px 30px rgba(0, 208, 255, 0.3);">
                
                <!-- Header -->
                <div style="background: rgba(0, 208, 255, 0.1); padding: 40px 30px; text-align: center; border-bottom: 2px solid rgba(0, 208, 255, 0.2);">
                    <h1 style="color: #00d0ff; margin: 0; font-size: 2.5em; font-weight: bold; text-shadow: 0 0 10px rgba(0, 208, 255, 0.5);">
                        DeepDiagnose
                    </h1>
                    <p style="color: #a0c4ff; margin: 10px 0 0 0; font-size: 1.1em;">
                        Welcome to the Future of Medical Diagnostics
                    </p>
                </div>
                
                <!-- Content -->
                <div style="padding: 40px 30px;">
                    <h2 style="color: #00d0ff; text-align: center; margin: 0 0 20px 0; font-size: 1.8em;">
                        🎉 Almost There!
                    </h2>
                    
                    <p style="color: #e2e8f0; font-size: 1.1em; line-height: 1.6; text-align: center; margin: 0 0 30px 0;">
                        Thank you for joining DeepDiagnose! To complete your account setup and start analyzing medical scans with AI, please verify your email address using the code below:
                    </p>
                    
                    <!-- OTP Code Box -->
                    <div style="background: rgba(0, 208, 255, 0.15); border: 2px solid #00d0ff; padding: 30px; text-align: center; border-radius: 15px; margin: 30px 0; position: relative; overflow: hidden;">
                        <div style="position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle, rgba(0, 208, 255, 0.1) 0%, transparent 70%); animation: pulse 2s infinite;"></div>
                        <p style="color: #a0c4ff; margin: 0 0 10px 0; font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; position: relative;">
                            Your Verification Code
                        </p>
                        <h1 style="font-family: 'Courier New', monospace; font-size: 3em; letter-spacing: 8px; color: #00d0ff; margin: 0; font-weight: bold; text-shadow: 0 0 15px rgba(0, 208, 255, 0.8); position: relative;">
                            {otp}
                        </h1>
                        <p style="color: #ff6b6b; margin: 15px 0 0 0; font-size: 0.9em; font-weight: bold; position: relative;">
                            ⏰ Expires in {settings.OTP_EXPIRY_MINUTES} minutes
                        </p>
                    </div>
                    
                    <!-- Instructions -->
                    <div style="background: rgba(255, 255, 255, 0.05); border-radius: 10px; padding: 25px; margin: 30px 0;">
                        <h3 style="color: #00d0ff; margin: 0 0 15px 0; font-size: 1.2em;">
                            📋 Next Steps:
                        </h3>
                        <ol style="color: #e2e8f0; margin: 0; padding-left: 20px; line-height: 1.8;">
                            <li>Return to the DeepDiagnose signup page</li>
                            <li>Enter the 6-digit code above</li>
                            <li>Start uploading and analyzing medical scans!</li>
                        </ol>
                    </div>
                    
                    <!-- Security Note -->
                    <div style="background: rgba(255, 107, 107, 0.1); border: 1px solid rgba(255, 107, 107, 0.3); border-radius: 8px; padding: 20px; margin: 30px 0;">
                        <p style="color: #ff6b6b; margin: 0; font-size: 0.95em; line-height: 1.5;">
                            🔒 <strong>Security Note:</strong> Never share this code with anyone. DeepDiagnose will never ask for your verification code via phone or social media. If you didn't create an account, please ignore this email.
                        </p>
                    </div>
                </div>
                
                <!-- Footer -->
                <div style="background: rgba(0, 0, 0, 0.2); padding: 30px; text-align: center; border-top: 1px solid rgba(0, 208, 255, 0.2);">
                    <p style="color: #718096; margin: 0 0 10px 0; font-size: 0.9em;">
                        Need help? Contact our support team at support@deepdiagnose.com
                    </p>
                    <p style="color: #4a5568; margin: 0; font-size: 0.8em;">
                        © 2025 DeepDiagnose. Revolutionizing Medical Diagnostics with AI.
                    </p>
                </div>
            </div>
            
            <!-- Add some CSS animation for the pulse effect -->
            <style>
                @keyframes pulse {{
                    0% {{ transform: scale(1); opacity: 0.8; }}
                    50% {{ transform: scale(1.05); opacity: 0.4; }}
                    100% {{ transform: scale(1); opacity: 0.8; }}
                }}
            </style>
        </body>
        </html>
        """
        
        # Create SendGrid mail object
        message = Mail(
            from_email=settings.SMTP_FROM_EMAIL,
            to_emails=email,
            subject="🔐 DeepDiagnose - Verify Your Account",
            html_content=html_content
        )
        
        # Initialize SendGrid client
        sg = SendGridAPIClient(api_key=settings.SMTP_PASSWORD)
        
        # Send email
        response = sg.send(message)
        
        # Check response status
        if response.status_code == 202:
            logger.info(f"OTP verification email sent to {email}")
        else:
            logger.error(f"SendGrid error: {response.status_code} - {response.body.decode('utf-8')}")
            
    except Exception as e:
        logger.exception(f"Failed to send OTP email to {email}: {str(e)}")

def send_reset_email(email: str, code: str):
    """
    Send password reset email using SendGrid (existing function)
    """
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