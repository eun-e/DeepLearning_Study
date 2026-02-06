class LSTMGateExplainer:
  def explain_gates(self, important_info, current_input, memory):
    forget_decision = self.forget_gate(memory, current_input)
    remember_decision = self.input_gate(current_input)
    output_decision = self.output_gate(memeory, current_input)

  def forget_gate(self, memory, current):
    if "날씨" in current or "점심" in current:
      return "일상적인 정보"
    return "중요 정보 유지"

  def input_gate(self, current):
    if "비밀번호" in current or "중요" in current:
      return "핵심 정보"
    return "일반 정보"

  def output_gate(self, memory, current):
    if "금고" in current:
      return "비밀번호 관련 정보"
    return "일반 응답"
