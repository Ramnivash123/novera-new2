"""
Email service for sending emails using SMTP.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional
from loguru import logger

from app.core.config import settings


class EmailService:
    """Service for sending emails via SMTP."""

    def __init__(self):
        self.smtp_host = settings.smtp_host
        self.smtp_port = settings.smtp_port
        self.smtp_user = settings.smtp_user
        self.smtp_password = settings.smtp_password
        self.from_email = settings.smtp_from_email
        self.from_name = settings.smtp_from_name
        self.use_tls = settings.smtp_use_tls

    def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """
        Send an email.

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML content of the email
            text_content: Plain text content (optional)

        Returns:
            True if email sent successfully
        """
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email

            if text_content:
                part1 = MIMEText(text_content, 'plain')
                msg.attach(part1)

            part2 = MIMEText(html_content, 'html')
            msg.attach(part2)

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                
                server.login(self.smtp_user, self.smtp_password)
                server.send_message(msg)

            logger.info(f"Email sent successfully to {to_email}")
            return True

        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {str(e)}")
            return False

    def send_password_reset_email(
        self,
        to_email: str,
        reset_token: str,
        username: str
    ) -> bool:
        """
        Send password reset email with reset link.

        Args:
            to_email: User's email address
            reset_token: Password reset token
            username: User's username

        Returns:
            True if email sent successfully
        """
        reset_link = f"{settings.frontend_url}/reset-password?token={reset_token}"
        
        # Logo URL - you can change this to your hosted logo URL
        logo_url = f"{settings.frontend_url}/Novera.jpg"

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Reset Your Password</title>
        </head>
        <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f5f7fa;">
            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #f5f7fa; padding: 40px 20px;">
                <tr>
                    <td align="center">
                        <!-- Main Container -->
                        <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="600" style="max-width: 600px; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);">
                            
                            <!-- Header -->
                            <tr>
                                <td style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); padding: 40px 30px; text-align: center;">
                                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                        <tr>
                                            <td align="center">
                                                <!-- Logo -->
                                                <img src="{logo_url}" alt="Novera AI" style="width: 120px; height: auto; margin-bottom: 20px; display: block; margin-left: auto; margin-right: auto;" />
                                            </td>
                                        </tr>
                                        <tr>
                                            <td align="center">
                                                <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 600; letter-spacing: -0.5px;">Password Reset Request</h1>
                                            </td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
                            
                            <!-- Content -->
                            <tr>
                                <td style="padding: 40px 30px;">
                                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                        <tr>
                                            <td>
                                                <p style="margin: 0 0 16px 0; color: #1f2937; font-size: 16px; line-height: 24px;">
                                                    Hello <strong>{username}</strong>,
                                                </p>
                                                <p style="margin: 0 0 24px 0; color: #4b5563; font-size: 15px; line-height: 22px;">
                                                    We received a request to reset the password for your Novera AI account. If you made this request, click the button below to reset your password.
                                                </p>
                                            </td>
                                        </tr>
                                        
                                        <!-- CTA Button -->
                                        <tr>
                                            <td align="center" style="padding: 30px 0;">
                                                <a href="{reset_link}" style="display: inline-block; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: #ffffff; text-decoration: none; font-size: 16px; font-weight: 600; padding: 14px 40px; border-radius: 8px; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);">
                                                    Reset Password
                                                </a>
                                            </td>
                                        </tr>
                                        
                                        <!-- Alternative Link -->
                                        <tr>
                                            <td>
                                                <p style="margin: 0 0 12px 0; color: #6b7280; font-size: 13px; line-height: 20px;">
                                                    Or copy and paste this link into your browser:
                                                </p>
                                                <p style="margin: 0 0 24px 0; padding: 12px; background-color: #f3f4f6; border-radius: 6px; word-break: break-all; font-size: 13px; color: #3b82f6; border: 1px solid #e5e7eb;">
                                                    {reset_link}
                                                </p>
                                            </td>
                                        </tr>
                                        
                                        <!-- Security Notice -->
                                        <tr>
                                            <td>
                                                <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #fef3c7; border-left: 4px solid #f59e0b; border-radius: 6px; padding: 16px;">
                                                    <tr>
                                                        <td>
                                                            <p style="margin: 0 0 12px 0; color: #92400e; font-size: 14px; font-weight: 600;">
                                                                ⚠️ Security Notice
                                                            </p>
                                                            <ul style="margin: 0; padding-left: 20px; color: #b45309; font-size: 13px; line-height: 20px;">
                                                                <li style="margin-bottom: 6px;">This link will expire in <strong>15 minutes</strong></li>
                                                                <li style="margin-bottom: 6px;">If you didn't request this, please ignore this email</li>
                                                                <li style="margin-bottom: 0;">Never share this link with anyone</li>
                                                            </ul>
                                                        </td>
                                                    </tr>
                                                </table>
                                            </td>
                                        </tr>
                                        
                                        <!-- Footer Text -->
                                        <tr>
                                            <td style="padding-top: 32px;">
                                                <p style="margin: 0; color: #6b7280; font-size: 14px; line-height: 20px;">
                                                    If you have any questions or need assistance, please contact our support team.
                                                </p>
                                                <p style="margin: 16px 0 24px 0; color: #1f2937; font-size: 14px; line-height: 20px;">
                                                    Best regards,<br>
                                                    <strong style="color: #3b82f6;">The Novera AI Team</strong>
                                                </p>
                                                
                                                <!-- Bottom Logo -->
                                                <div style="text-align: center; margin-top: 24px; padding-top: 24px; border-top: 1px solid #e5e7eb;">
                                                    <img src="{logo_url}" alt="Novera AI" style="width: 100px; height: auto; opacity: 0.6;" />
                                                </div>
                                            </td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
                            
                            <!-- Footer -->
                            <tr>
                                <td style="padding: 30px; background-color: #f9fafb; text-align: center;">
                                    <p style="margin: 0 0 8px 0; color: #9ca3af; font-size: 12px; line-height: 18px;">
                                        This is an automated message, please do not reply to this email.
                                    </p>
                                    <p style="margin: 0; color: #9ca3af; font-size: 12px; line-height: 18px;">
                                        &copy; 2025 Novera AI. All rights reserved.
                                    </p>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """

        text_content = f"""
        Password Reset Request - Novera AI
        
        Hello {username},
        
        We received a request to reset the password for your Novera AI account.
        
        Click the link below to reset your password:
        {reset_link}
        
        SECURITY NOTICE:
        - This link will expire in 15 minutes
        - If you didn't request this, please ignore this email
        - Never share this link with anyone
        
        If you have any questions or need assistance, please contact our support team.
        
        Best regards,
        The Novera AI Team
        
        ---
        This is an automated message, please do not reply to this email.
        © 2025 Novera AI. All rights reserved.
        """

        return self.send_email(
            to_email=to_email,
            subject="Reset Your Password - Novera AI",
            html_content=html_content,
            text_content=text_content
        )

    def send_verification_email(
        self,
        to_email: str,
        verification_token: str,
        username: str
    ) -> bool:
        """
        Send email verification email with activation link.

        Args:
            to_email: User's email address
            verification_token: Email verification token
            username: User's username

        Returns:
            True if email sent successfully
        """
        verification_link = f"{settings.frontend_url}/verify-email?token={verification_token}"
        
        # Logo URL - you can change this to your hosted logo URL
        logo_url = f"{settings.frontend_url}/Novera.jpg"

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Verify Your Email</title>
        </head>
        <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f5f7fa;">
            <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #f5f7fa; padding: 40px 20px;">
                <tr>
                    <td align="center">
                        <!-- Main Container -->
                        <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="600" style="max-width: 600px; background-color: #ffffff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);">
                            
                            <!-- Header -->
                            <tr>
                                <td style="background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); padding: 40px 30px; text-align: center;">
                                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                        <tr>
                                            <td align="center">
                                                <!-- Top Logo -->
                                                <img src="{logo_url}" alt="Novera AI" style="width: 120px; height: auto; margin-bottom: 20px; display: block; margin-left: auto; margin-right: auto;" />
                                            </td>
                                        </tr>
                                        <tr>
                                            <td align="center">
                                                <h1 style="margin: 0; color: #ffffff; font-size: 28px; font-weight: 600; letter-spacing: -0.5px;">Welcome to Novera AI!</h1>
                                            </td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
                            
                            <!-- Content -->
                            <tr>
                                <td style="padding: 40px 30px;">
                                    <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%">
                                        <!-- Welcome Message -->
                                        <tr>
                                            <td>
                                                <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); border-radius: 8px; padding: 20px; margin-bottom: 24px;">
                                                    <tr>
                                                        <td>
                                                            <p style="margin: 0 0 8px 0; color: #1e40af; font-size: 16px; font-weight: 600;">
                                                                Hello {username}! 👋
                                                            </p>
                                                            <p style="margin: 0; color: #1e3a8a; font-size: 14px; line-height: 20px;">
                                                                Thanks for joining Novera AI Knowledge Assistant. You're just one step away from unlocking the power of AI-driven insights!
                                                            </p>
                                                        </td>
                                                    </tr>
                                                </table>
                                            </td>
                                        </tr>
                                        
                                        <tr>
                                            <td>
                                                <p style="margin: 0 0 24px 0; color: #4b5563; font-size: 15px; line-height: 22px;">
                                                    To get started and access all features, please verify your email address by clicking the button below:
                                                </p>
                                            </td>
                                        </tr>
                                        
                                        <!-- CTA Button -->
                                        <tr>
                                            <td align="center" style="padding: 30px 0;">
                                                <a href="{verification_link}" style="display: inline-block; background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); color: #ffffff; text-decoration: none; font-size: 16px; font-weight: 600; padding: 14px 40px; border-radius: 8px; box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);">
                                                    Verify Email Address
                                                </a>
                                            </td>
                                        </tr>
                                        
                                        <!-- Info Box -->
                                        <tr>
                                            <td>
                                                <table role="presentation" cellspacing="0" cellpadding="0" border="0" width="100%" style="background-color: #eff6ff; border: 1px solid #bfdbfe; border-radius: 6px; padding: 16px; margin-top: 24px;">
                                                    <tr>
                                                        <td>
                                                            <p style="margin: 0 0 12px 0; color: #1e40af; font-size: 14px; font-weight: 600;">
                                                                📌 Important Information
                                                            </p>
                                                            <ul style="margin: 0; padding-left: 20px; color: #1e3a8a; font-size: 13px; line-height: 20px;">
                                                                <li style="margin-bottom: 6px;">This verification link expires in <strong>24 hours</strong></li>
                                                                <li style="margin-bottom: 6px;">You can request a new link if this one expires</li>
                                                                <li style="margin-bottom: 0;">Some features are limited until your email is verified</li>
                                                            </ul>
                                                        </td>
                                                    </tr>
                                                </table>
                                            </td>
                                        </tr>
                                        
                                        <!-- Footer Text -->
                                        <tr>
                                            <td style="padding-top: 32px;">
                                                <p style="margin: 0 0 16px 0; color: #6b7280; font-size: 14px; line-height: 20px;">
                                                    If you didn't create an account with Novera AI, you can safely ignore this email.
                                                </p>
                                                <p style="margin: 0 0 24px 0; color: #1f2937; font-size: 14px; line-height: 20px;">
                                                    Best regards,<br>
                                                    <strong style="color: #3b82f6;">The Novera AI Team</strong>
                                                </p>
                                                
                                                <!-- Bottom Logo -->
                                                <div style="text-align: center; margin-top: 24px; padding-top: 24px; border-top: 1px solid #e5e7eb;">
                                                    <img src="{logo_url}" alt="Novera AI" style="width: 100px; height: auto; opacity: 0.6;" />
                                                </div>
                                            </td>
                                        </tr>
                                    </table>
                                </td>
                            </tr>
                            
                            <!-- Footer -->
                            <tr>
                                <td style="padding: 30px; background-color: #f9fafb; text-align: center;">
                                    <p style="margin: 0 0 8px 0; color: #9ca3af; font-size: 12px; line-height: 18px;">
                                        This is an automated message, please do not reply to this email.
                                    </p>
                                    <p style="margin: 0; color: #9ca3af; font-size: 12px; line-height: 18px;">
                                        &copy; 2025 Novera AI. All rights reserved.
                                    </p>
                                </td>
                            </tr>
                        </table>
                    </td>
                </tr>
            </table>
        </body>
        </html>
        """

        text_content = f"""
        Welcome to Novera AI!
        
        Hello {username}! 👋
        
        Thanks for joining Novera AI Knowledge Assistant. You're just one step away from unlocking the power of AI-driven insights!
        
        To get started and access all features, please verify your email address by clicking the link below:
        {verification_link}
        
        IMPORTANT INFORMATION:
        - This verification link expires in 24 hours
        - You can request a new link if this one expires
        - Some features are limited until your email is verified
        
        If you didn't create an account with Novera AI, you can safely ignore this email.
        
        Best regards,
        The Novera AI Team
        
        ---
        This is an automated message, please do not reply to this email.
        © 2025 Novera AI. All rights reserved.
        """

        return self.send_email(
            to_email=to_email,
            subject="Verify Your Email - Novera AI",
            html_content=html_content,
            text_content=text_content
        )


email_service = EmailService()

__all__ = ['EmailService', 'email_service']