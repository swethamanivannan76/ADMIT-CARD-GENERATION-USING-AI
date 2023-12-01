class TextScramble {
    constructor(el) {
      this.el = el;
      this.chars = '!<>-_\\/[]{}—=+*^?#________';
      this.update = this.update.bind(this);
    }
    
    setText(newText) {
      const oldText = this.el.innerText;
      const length = Math.max(oldText.length, newText.length);
      const promise = new Promise(resolve => (this.resolve = resolve));
      this.queue = [];
      
      for (let i = 0; i < length; i++) {
        const from = oldText[i] || '';
        const to = newText[i] || '';
        const start = Math.floor(Math.random() * 40);
        const end = start + Math.floor(Math.random() * 40);
        this.queue.push({ from, to, start, end });
      }
      
      cancelAnimationFrame(this.frameRequest);
      this.frame = 0;
      this.update();
      return promise;
    }
    
    update() {
      let output = '';
      let complete = 0;
      
      for (let i = 0, n = this.queue.length; i < n; i++) {
        let { from, to, start, end, char } = this.queue[i];
        
        if (this.frame >= end) {
          complete++;
          output += to;
        } else if (this.frame >= start) {
          if (!char || Math.random() < 0.28) {
            char = this.randomChar();
            this.queue[i].char = char;
          }
          output += `<span class="dud">${char}</span>`;
        } else {
          output += from;
        }
      }
      
      this.el.innerHTML = output;
      
      if (complete === this.queue.length) {
        this.resolve();
      } else {
        this.frameRequest = requestAnimationFrame(this.update);
        this.frame++;
      }
    }
    
    randomChar() {
      return this.chars[Math.floor(Math.random() * this.chars.length)];
    }
  }
  
  const phrases = [
    'project',
    'done by',
    'Tech Hunters', 
    'team members',
    'Harishh',
    'Swetha',
    'Dinesh',
    'Sridevi',
    'Gokul',
    'Arigato!!'
    
  ];
  
  const el = document.querySelector('.text-change');
  const fx = new TextScramble(el);
  
  let counter = 0;
  
  const next = () => {
    fx.setText(phrases[counter]).then(() => {
      setTimeout(next, 1000);
    });
    counter = (counter + 1) % phrases.length;
  };
  
  next();
  
  function sendOtp() {
    var phoneNumber = document.querySelector('input[name="phone_number"]').value;

    // Send an asynchronous request to send OTP
    fetch('/send_otp', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ phone_number: phoneNumber })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Show the OTP input field
            document.getElementById('otp-section').style.display = 'block';
            // Hide the "Get OTP" button
            document.getElementById('otp-button').style.display = 'none';
            // Show the "Generate Admit Card" button
            document.getElementById('generate-button').style.display = 'block';
        } else {
            alert('Failed to send OTP. Please try again.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

  
  
  
  
  